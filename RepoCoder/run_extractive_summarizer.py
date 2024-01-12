from utils import Tools, FilePathBuilder
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import remove_comments_and_docstrings, tree_to_token_index, index_to_code_token, tree_to_variable_index
from tree_sitter import Language, Parser

from merge_score import get_n_chunk

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
    
    
def get_extractive_summary(code, parser, matching_candidates, omitting_candidates=[]):
    tree = parser[0].parse(bytes(code,'utf8'))    
    root_node = tree.root_node  
    tokens_index=tree_to_token_index(root_node)

    _code=code.split('\n')
    code_tokens=[index_to_code_token(x,_code) for x in tokens_index]
    tokens_type=tree_to_token_type(root_node)

    extractive_summary = []
    for token, token_type in zip(code_tokens, tokens_type):
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
#                 print(f"{token}\t{token_type}")
                pass

    return " ".join(extractive_summary)


lang = 'python'
_parser = get_parser()
parser = _parser[lang]
    
call_att_identifier =  ('call', 'attribute', 'identifier')
call_identifier =  ('call', 'identifier')
keyword_argument =  ('call', 'argument_list', 'keyword_argument', 'identifier')


if __name__ == '__main__':
    base_path='results'
    benchmark = 'short_line' # 'short_api'
    model_name = 'codegen-2B-mono' # 'deepseek-coder-6.7b-instruct' #'codegen-6B-mono' # 
    chunk_size = 1
    rg_predictions = 'rg-one-gram-ws-20-ss-2'
    rg_prompts = Tools.get_prompt(f"prompts/{benchmark}/{model_name}/{rg_predictions}.jsonl")
    option = 'extractive_summary_omission'
    predictions = f'{option}-one-gram-ws-20-ss-2'
    iteration = 0
    matching_candidates = [] # [call_att_identifier, call_identifier, keyword_argument]
    omitting_candidates = [call_att_identifier, call_identifier, keyword_argument]
    

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
    for v in samples['predictions']:
        _summary_list = []
        for code in v:
            _summary = get_extractive_summary(code, parser, 
                                              matching_candidates,
                                              omitting_candidates)
            _summary_list.append(_summary)
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