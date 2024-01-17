from utils import Tools, FilePathBuilder
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import remove_comments_and_docstrings, tree_to_token_index, index_to_code_token, tree_to_variable_index
from tree_sitter import Language, Parser

from merge_score import get_n_chunk

# PYTHON DEPENDENCY
# https://www.w3schools.com/python/python_datatypes.asp
DEFAULT_TYPES = ['str', 
                 'int', 'float', 'complex', 
                 'list', 'tuple', 'range', 
                 'dict',
                 'set', 'frozenset',
                 'bool',
                 'bytes', 'bytearray', 'memoryview',
                 'NoneType']

# https://docs.python.org/3/library/functions.html
BUILT_IN_FUNCTIONS = ["abs", "aiter", "all", "anext", "any", "ascii", "bin", "bool", "breakpoint", 
                      "bytearray", "bytes", "callable", "chr", "classmethod", "compile", "complex", 
                      "delattr", "dict", "dir", "divmod", "enumerate", "eval", "exec", "filter", 
                      "float", "format", "frozenset", "getattr", "globals", "hasattr", "hash", "help", 
                      "hex", "id", "input", "int", "isinstance", "issubclass", "iter", "len", "list", 
                      "locals", "map", "max", "memoryview", "min", "next", "object", "oct", "open", 
                      "ord", "pow", "print", "property", "range", "repr", "reversed", "round", "set", 
                      "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", 
                      "type", "vars", "zip", "__import__"]

def get_parser():
    dfg_function={
        'python':DFG_python,
        'java':DFG_java,
        'ruby':DFG_ruby,
        'go':DFG_go,
        'php':DFG_php,
        'javascript':DFG_javascript
    }
    _parser = {}
    for lang in dfg_function.keys():
        LANGUAGE = Language('parser/my-languages.so', lang)
        p = Parser()
        p.set_language(LANGUAGE) 
        _parser[lang] = [p,dfg_function[lang]]
    return _parser


def tree_to_token_type(root_node, types=tuple()):
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        return [types+(root_node.type,)]
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_token_type(child, types+(root_node.type,))
        return code_tokens
    
    
def get_extractive_summary(code, parser, matching_candidates, 
                           omitting_candidates=[], window=5, option='node', 
                           no_built_in_ids=True):
    assert option in ['node', 'node_sequence']
    tree = parser[0].parse(bytes(code,'utf8'))    
    root_node = tree.root_node  
    tokens_index=tree_to_token_index(root_node)

    _code=code.split('\n')
    code_tokens=[index_to_code_token(x,_code) for x in tokens_index]
    tokens_type=tree_to_token_type(root_node)

    extractive_summary = []
    log = []
    
    if option == 'node':
        for token, token_type in zip(code_tokens, tokens_type):
            if token_type[-1] != 'identifier':
                continue
            token_type = token_type[-window:]
            if 'ERROR' in token_type:
                continue
            log.append((token, token_type))
    #         print(f"{token}\t{token_type}")
            omit_flag = False
            # Omission 
            for cand in omitting_candidates:
                if cand in token_type:                
                    omit_flag = True
                    break
            if omit_flag:
                continue

            # Call identifier
            for cand in matching_candidates:
                if cand in token_type:   
                    # Ignore built-in functions / default types 
                    if no_built_in_ids and token in DEFAULT_TYPES+BUILT_IN_FUNCTIONS:
                        continue
                    extractive_summary.append(token)
                else:
                    pass

    # stale version
    elif option == 'node_sequence': 
        print(f'warning: {option} is a stale version!')
        for token, token_type in zip(code_tokens, tokens_type):
            log.append((token, token_type))
    #         print(f"{token}\t{token_type}")
            omit_flag = False
            # Omission 
            for cand in omitting_candidates:
                tail = token_type[-len(cand):]
                if tail == cand:
                    omit_flag = True
                    break
            if omit_flag:
                continue

            # Call identifier
            for cand in matching_candidates:
                tail = token_type[-len(cand):]
                if tail == cand:
                    extractive_summary.append(token)
                else:
                    pass
                
    return " ".join(extractive_summary), log


def run(parser,
        base_path='results',
        benchmark='short_line',
        model_name='codegen-2B-mono',
        chunk_size=1,
        rg_predictions='rg-one-gram-ws-20-ss-2',
        option='extractive_summary_identifier',
        iteration=0,
        matching_candidates=['identifier'], 
        omitting_candidates=[],
        _option='node',
        no_built_in_ids=True):
    
    rg_prompts = Tools.get_prompt(f"prompts/{benchmark}/{model_name}/{rg_predictions}.jsonl")
    predictions=f'{option}-one-gram-ws-20-ss-2'
    
    n_chunk = get_n_chunk(base_path, benchmark, model_name, option='rg')
    samples = None
    for chunk_idx in range(n_chunk):
        tmp_samples = Tools.load_pickle(f"results/{benchmark}/rg_{model_name}_{chunk_idx}.pkl")        
        if samples is None:
            samples = tmp_samples
        else:
            for key, val in tmp_samples.items():
                samples[key].extend(val)
                
    extractive_summary = []
    log = []
    for v in samples['predictions']:
        _summary_list = []
        for code in v:
            _summary, _log = get_extractive_summary(code, parser, 
                                              matching_candidates,
                                              omitting_candidates,
                                              option=_option,
                                              no_built_in_ids=no_built_in_ids)
            _summary_list.append(_summary)
            log.extend(_log)
        extractive_summary.append(_summary_list)
        
        
    sample_dict_list = []
    for i in range(len(rg_prompts)):
        meta_dict = {key: rg_prompts[i]['metadata'][key] 
                              for key in ["task_id", "fpath_tuple", \
                                          "line_no", "context_start_lineno"]}

        sample_dict = {'metadata': meta_dict, 
                           'choices': [{'text': v} for v in extractive_summary[i]]}
        sample_dict_list.append(sample_dict)

    save_path = f'predictions/{benchmark}/{model_name}/{predictions}_samples.{iteration}.jsonl'
    FilePathBuilder.make_needed_dir(save_path)
    Tools.dump_jsonl(sample_dict_list, save_path)
    
    
    return samples, extractive_summary, sample_dict_list, log



if __name__ == '__main__':
    lang = 'python'
    _parser = get_parser()
    parser = _parser[lang]

    model_name_list = ['codegen-2B-mono', 'codegen-6B-mono', 
                       'deepseek-coder-6.7b-base',
                       'deepseek-coder-6.7b-instruct']
    short_benchmarks = ['short_line', 'short_api']
    long_benchmarks = ['random_line', 'random_api']
    
    #### stale
    call_att_identifier =  ('call', 'attribute', 'identifier')
    call_identifier =  ('call', 'identifier')
    keyword_argument =  ('call', 'argument_list', 'keyword_argument', 'identifier')
    identifier = ('identifier',)
    
    entities = ['import_from_statement', 'import_statement', 'call', 'keyword_argument', 'type']
    non_entities = ['decorated_definition', 'function_definition', 
                    'raise_statement', 'class_definition', 'subscript', 'lambda']
    
    
    option = 'extractive_summary_non_entity_no_built_in_ids' # 'extractive_summary' # 'extractive_summary_identifier'
    predictions = f'{option}-one-gram-ws-20-ss-2'

    #### Stale
    if option == 'extractive_summary': 
        matching_candidates = [call_att_identifier, call_identifier, keyword_argument]
        omitting_candidates = []
        _option = 'node_sequence'
        no_built_in_ids = False
    elif option == 'extractive_summary_identifier': 
        matching_candidates = ['identifier'] # [call_att_identifier, call_identifier, keyword_argument]
        omitting_candidates = [] # [call_att_identifier, call_identifier, keyword_argument]
        _option = 'node'
        no_built_in_ids = False        
    elif option == 'extractive_summary_entity':
        matching_candidates = entities
        omitting_candidates = non_entities
        _option = 'node'
        no_built_in_ids = True
    elif option == 'extractive_summary_non_entity':
        matching_candidates = ['identifier']
        omitting_candidates = entities
        _option = 'node'
        no_built_in_ids = False
    elif option == 'extractive_summary_non_entity_no_built_in_ids':
        matching_candidates = ['identifier']
        omitting_candidates = entities
        _option = 'node'
        no_built_in_ids = True
    
    print(f"{option=}")
    log = []
    for model_name in model_name_list:
        if model_name.split('-')[0] == 'codegen':
            _benchmarks = short_benchmarks
        elif model_name.split('-')[0] == 'deepseek':
            _benchmarks = long_benchmarks
        else:
            raise NotImplementedError
        
        print(f"{model_name=}")
        for benchmark in _benchmarks:    
            print(f"{benchmark=}")            
            samples, extractive_summary, sample_dict_list, _log = run(parser, benchmark=benchmark, 
                                                                      model_name=model_name, 
                                                                      option=option,
                                                                      matching_candidates=matching_candidates,
                                                                      omitting_candidates=omitting_candidates,
                                                                      _option=_option,
                                                                      no_built_in_ids=no_built_in_ids
                                                                      )
            log.extend(_log)
        print()