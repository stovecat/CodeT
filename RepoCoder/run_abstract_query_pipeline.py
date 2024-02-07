import copy
import tqdm
import itertools
import functools

import numpy as np
from scipy.stats import spearmanr

from concurrent.futures import as_completed, ProcessPoolExecutor


from utils import *
from inference import splitter

from search_code import SimilarityScore, CodeSearchWorker, CodeSearchWrapper
from compute_score import compute_new_ES

from make_window import RepoWindowMaker, PredictionWindowMaker, MakeWindowWrapper

from build_vector import BuildVectorWrapper

from build_prompt import BuildPromptWrapper, PromptBuilder

from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import remove_comments_and_docstrings, tree_to_token_index, index_to_code_token, tree_to_variable_index
from tree_sitter import Language, Parser



def get_parser():
    #################################################
    # Return AST parsers
    #################################################
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

_parser = get_parser()

lang = 'python'
parser = _parser[lang]


def remove_multiline_paranthesis(lines):
    #################################################
    # Flag is for debug purpose
    #################################################
    flag = False
    import copy
    lines = [l for l in lines if l.replace(" ", "")[0] != '#']
    _lines = copy.deepcopy(lines)
    len_lines = len(lines)
    for i in range(len_lines):
        # from back
        i = len_lines - 1 - i
        line = lines[i]
        n_opened_parenthesis = line.count('(')
        n_closed_parenthesis = line.count(')')
        #####################################################
        # When a multi-line paranthesis is closed
        #####################################################
        if (i > 0 and n_opened_parenthesis < n_closed_parenthesis):
            filler = ""
            # Fill blank for multiline arguments
            if lines[i-1][-1] == ',':
                filler = " "
            lines[i-1] += filler+lines[i]
            lines.pop(i)
            flag = True
        #####################################################
        # When a multi-line paranthesis is unclosed
        #####################################################
        elif i==0 and n_opened_parenthesis > n_closed_parenthesis and len(lines) > 1:
            filler = ""
            # Fill blank for multiline arguments
            if lines[i][-1] == ',':
                filler = " "
            lines = [filler.join(lines)]
            flag = True            
#     if flag:
#         print(_lines)
#         print("-"*40)
#         print(lines)
#         print("="*40)
    return lines, flag


def get_nodes(code):
    #################################################
    # Return a sequence of meaningful AST nodes
    #################################################
    tree = parser[0].parse(bytes(code,'utf8'))
    root_node = tree.root_node

    _nodes = root_node.sexp()
    # A list of meaningless nodes
    nodes = ['(', ')', ':', 'identifier', 'left', 'right', 'expression_statement', 'ERROR', 'module', 'body', 'argument_list', 'dotted_name', 'block']
    for node in nodes:
        _nodes = _nodes.replace(node, '')
        
    return tuple([v for v in _nodes.split(" ") if v != ""][:5])


def get_ast_sequence(code, lang):
    ############################################################
    # Return an abstract code line consists of the AST nodes
    ############################################################
    try:
        code = remove_comments_and_docstrings(code, lang)
    except Exception as e:
        pass
    
    _lines = [line.strip() for line in code.split(splitter)[-1].splitlines() if line.strip()]
    lines, _ = remove_multiline_paranthesis(_lines)
        
    ast_sequences = [" ".join(get_nodes(l)) for l in lines]
    
    return lines, ast_sequences




class RepoAbstractWindowMaker(RepoWindowMaker):
    def __init__(self, repo, window_size, slice_size):
        super().__init__(repo, window_size, slice_size)
    
    def build_windows(self):
        input_path = FilePathBuilder.repo_windows_path(self.repo, self.window_size, self.slice_size)
        #################################################
        # Load the original window
        #################################################
        repos = Tools.load_pickle(input_path)
        abs_repos = []
        for d in repos:
            #################################################
            # Use the abstract code lines as the context
            #################################################
            sample_lines, ast_sequences = get_ast_sequence(d['context'], lang)
            abs_repos.append({
                'context': '\n'.join(ast_sequences),
                'metadata': d['metadata']
            })
        print(f'build {len(abs_repos)} windows for {self.repo} with window size {self.window_size} and slice {self.slice_size}')
        output_path = FilePathBuilder.repo_abs_windows_path(self.repo, self.window_size, self.slice_size)
        Tools.dump_pickle(abs_repos, output_path)


class PredictionAbstractWindowMaker(PredictionWindowMaker):
    def __init__(self, repo, window_size, prediction_path, window_path_builder, option):
        #################################################
        # TBD: categorize options
        #################################################
        self.option = option
        super().__init__(repo, window_size, prediction_path, window_path_builder)
    
    def build_window(self, type='centered'):
        code_windows = []
        delta_size = self.window_size // 2
        for prediction in self.predictions:
            if prediction['metadata']['task_id'].split('/')[0] != self.repo:
                continue
            fpath_tuple = tuple(prediction['metadata']['fpath_tuple'])
            line_no = prediction['metadata']['line_no']  # line_no in prediction file starts from 0
            original_code = self.source_code[fpath_tuple]
            
            ###################################################################
            # TBD: Include unfinished abstract code lines into the context
            ###################################################################
            if 'include_unfinished' in self.option:
                original_lines, original_ast_sequences = get_ast_sequence(original_code, lang)          
#             code_lines = original_code.splitlines()
            
            context_start_lineno = prediction['metadata']['context_start_lineno']
            start_line_no = max(context_start_lineno, line_no - delta_size)
            for sample in [prediction['choices'][i]['text'] for i in range(len(prediction['choices']))]:
                sample_lines, ast_sequences = get_ast_sequence(sample, lang)
                end_line_no = min(len(ast_sequences), line_no + self.window_size - delta_size)
    
                #################################################
                # Use the abstract code line as the context
                #################################################
                context = '\n'.join(ast_sequences)
                if 'include_unfinished' in self.option:
                    context = '\n'.join(original_ast_sequences)+'\n'+context
                code_windows.append({
                    'context': context,
                    'metadata': {
                        'fpath_tuple': fpath_tuple,
                        'line_no': line_no,  # line_no starts from 0
                        'prediction': sample,
                        'task_id': prediction['metadata']['task_id'],
                        'start_line_no': start_line_no,
                        'end_line_no': end_line_no,
                        'window_size': self.window_size,
                        'context_start_lineno': context_start_lineno,
                        'repo': self.repo
                    }
                })
        print(f'build {len(code_windows)} prediction windows for {self.repo} with window size {self.window_size}')
        output_path = self.window_path_builder(self.prediction_path, self.repo, self.window_size, self.option)
        Tools.dump_pickle(code_windows, output_path)
        
        
class MakeAbstractWindowWrapper(MakeWindowWrapper):
    def __init__(self, benchmark, repos, window_sizes, slice_sizes):
        super().__init__(benchmark, repos, window_sizes, slice_sizes)

    def window_for_abs_repo_files(self):
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            for repo in self.repos:
                repo_window_maker = RepoAbstractWindowMaker(repo, window_size, slice_size)
                repo_window_maker.build_windows()

        
    def window_for_abstract_prediction(self, mode, prediction_path_template, option):
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            prediction_path = prediction_path_template.format(window_size=window_size, slice_size=slice_size)
            for repo in self.repos:
                window_path_builder = functools.partial(FilePathBuilder.gen_first_abs_window_path, self.benchmark, mode)
                pred_window_maker = PredictionAbstractWindowMaker(repo, window_size, prediction_path, window_path_builder, option)
                pred_window_maker.build_window()


                
def make_abs_repo_window(repos, window_sizes, slice_sizes):
    ######################################################
    # Build candidates consist of abstract code lines 
    ######################################################
    worker = MakeAbstractWindowWrapper(None, repos, window_sizes, slice_sizes)
    worker.window_for_abs_repo_files()
    
    
class ASTSequence:
    ######################################################
    # AST sequence vector: string as is 
    # Scored by: Edit Similarity
    ######################################################
    def __init__(self, input_file):
        self.input_file = input_file

    def build(self):
        print(f'building AST sequence vector for {self.input_file}')
        futures = dict()
        lines = Tools.load_pickle(self.input_file)
        with ProcessPoolExecutor(max_workers=48) as executor:
            for line in lines:
                futures[executor.submit(Tools.tokenize, line['context'])] = line
        
            new_lines = []
            t = tqdm.tqdm(total=len(futures))
            for future in as_completed(futures):
                line = futures[future]
                tokenized = future.result()
                new_lines.append({
                    'context': line['context'],
                    'metadata': line['metadata'],
                    'data': [{'embedding': line['context']}]
                })
                tqdm.tqdm.update(t)
            output_file_path = FilePathBuilder.ast_sequence_vector_path(self.input_file)
            print(f"{output_file_path=}")
            Tools.dump_pickle(new_lines, output_file_path)
            
            

class BuildAbstractVectorWrapper(BuildVectorWrapper):
    def __init__(self, benchmark, vector_builder, repos, window_sizes, slice_sizes):
        super().__init__(benchmark, vector_builder, repos, window_sizes, slice_sizes)

    def vectorize_abstract_repo_windows(self):
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            for repo in self.repos:
                builder = self.vector_builder(
                    FilePathBuilder.repo_abs_windows_path(repo, window_size, slice_size)
                )
                builder.build()
        
    def vectorize_abstract_prediction_windows(self, mode, prediction_path_template, option):
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            prediction_path = prediction_path_template.format(window_size=window_size, slice_size=slice_size)
            for repo in self.repos:
                window_path = FilePathBuilder.gen_first_abs_window_path(
                    self.benchmark, mode, prediction_path, repo, window_size, option
                )
                builder = self.vector_builder(window_path)
                builder.build()
                
                
                
class SimilarityScore:
    @staticmethod
    def cosine_similarity(embedding_vec1, embedding_vec2):
        return 1 - scipy.spatial.distance.cosine(embedding_vec1, embedding_vec2)
    
    @staticmethod
    def jaccard_similarity(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / union

    @staticmethod
    def compute_new_ES(text1, text2):
        if text1.tolist() == '' or text2.tolist() == '':
            return 0.
        return compute_new_ES(text1.tolist(), [text2.tolist()], passk=1)
        
        

class CodeSearchASTWrapper(CodeSearchWrapper):
    def __init__(self, vectorizer, benchmark, repos, window_sizes, slice_sizes):
        super().__init__(vectorizer, benchmark, repos, window_sizes, slice_sizes)
        ###################################################################
        # Store all context for the prompt construction (max_top_k: 0)
        # a = [1, 2, 3, 4]
        # a[-0:]
        # >> [1, 2, 3, 4]
        ###################################################################
        self.max_top_k = 0 
        
        if vectorizer == 'ast-sequence':
            self.sim_scorer = SimilarityScore.compute_new_ES
            self.vector_path_builder = FilePathBuilder.ast_sequence_vector_path
    
    def _run_parallel(self, query_window_path_builder, prediction_path_template=None, option=''):
        workers = []
        for window_size in self.window_sizes:
            for slice_size in self.slice_sizes:
                for repo in self.repos:
                    if prediction_path_template:
                        query_window_path = query_window_path_builder(
                            prediction_path_template.format(window_size=window_size, slice_size=slice_size),
                            repo, window_size, option
                        )
                    else:
                        query_window_path = query_window_path_builder(repo, window_size)
                    query_line_path = self.vector_path_builder(query_window_path)
                    repo_window_path = FilePathBuilder.repo_abs_windows_path(repo, window_size, slice_size)
                    repo_embedding_path = self.vector_path_builder(repo_window_path)
                    output_path = FilePathBuilder.abs_retrieval_results_path(query_line_path, repo_embedding_path, self.max_top_k)
                    repo_embedding_lines = Tools.load_pickle(repo_embedding_path)
                    query_embedding_lines = Tools.load_pickle(query_line_path)
                    log_message = f'repo: {repo}, window: {window_size}, slice: {slice_size}  {self.vectorizer}, max_top_k: {self.max_top_k}'
                    worker = CodeSearchWorker(repo_embedding_lines, query_embedding_lines, output_path, self.sim_scorer, self.max_top_k, log_message)
                    workers.append(worker)
        # process pool
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(worker.run, ) for worker in workers}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                future.result()

    def search_ast_prediction(self, mode, prediction_path_template, option):
        query_line_path_temp = functools.partial(FilePathBuilder.gen_first_abs_window_path, self.benchmark, mode)
        self._run_parallel(query_line_path_temp, prediction_path_template, option)
        
        
class CodeSearchWrapper(CodeSearchWrapper):
    def __init__(self, vectorizer, benchmark, repos, window_sizes, slice_sizes):
        super().__init__(vectorizer, benchmark, repos, window_sizes, slice_sizes)
        ###################################################################
        # Store all context for the prompt construction (max_top_k: 0)
        # a = [1, 2, 3, 4]
        # a[-0:]
        # >> [1, 2, 3, 4]
        ###################################################################
        self.max_top_k = 0
        
        
        
def get_data_id(d):
    ################################
    # Return the id of problem
    ################################
    return d['metadata']['task_id']

def load_retrieval(path):
    ##########################################
    # Load and sort the retrieval results
    ##########################################
    return sorted(Tools.load_pickle(path), key=lambda x:get_data_id(x))

def get_cand_id(d):
    ##########################################
    # Return the id of candidate
    ##########################################
    _dict = d[0]['metadata'][0]
    return "/".join(_dict['fpath_tuple'])+'/'+str(_dict['line_no'])



def get_spearman_rank_corr(retrieval_scores, alpha=1.0, beta=0.0):
    ########################################################################################################################
    # Return the spearman's rank correlation between 1) and 2):
    # 1) the obtained ranking scores 
    # 2) the ranking scores measured by the edit similarity between each candidate and the ground truth code line (i.e. ES(cand, gt_code_line))
    ########################################################################################################################
    correlation_score = {'none': [], 'repocoder': [], 'abs': [], 'gt': [], 'mixed': []}
    correlation_p_val = {'none': [], 'repocoder': [], 'abs': [], 'gt': [], 'mixed': []}
    for i in range(len(retrieval_scores)):
        flatten_scores = {'none': [], 'repocoder': [], 'abs': [], 'gt': [], 'ES': [], 'mixed': []}
        for key in retrieval_scores[i].keys():
            for _key in retrieval_scores[i][key].keys():
                flatten_scores[_key].append(retrieval_scores[i][key][_key])
        for _key in flatten_scores.keys():
            if _key == 'ES':
                continue
            ####################
            # Mix the scores
            ####################
            if _key == 'mixed':
                corr, p_value = spearmanr([alpha*(v1) + (1-alpha)*(beta*v2+(1-beta)*v3) 
                                           for (v1, (v2,v3)) in zip(flatten_scores['none'], 
                                                  zip(flatten_scores['repocoder'],flatten_scores['abs']))], 
                                          flatten_scores['ES'])
            else:
                corr, p_value = spearmanr(flatten_scores[_key], flatten_scores['ES'])
#             sorted_results = sorted([(v1,v2) for v1, v2 in zip(flatten_scores[_key], flatten_scores['ES'])],
#                                     key=lambda x:x[0], reverse=True)[:topk]
#             corr, p_value = spearmanr([v[0] for v in sorted_results], [v[1] for v in sorted_results])
            if corr is not np.nan and p_value is not np.nan:
                correlation_score[_key].append(corr)
                correlation_p_val[_key].append(p_value)

    print(f"{alpha=},{beta=}")
    print("Correlation")
    for k,v in correlation_score.items():
        print(k, np.mean(v), f'(std={round(np.std(v),2)}, #examples p_val<0.01={sum([1 for v in correlation_p_val[k] if v<0.01])}/{len(correlation_p_val[k])})')
        
        
        



def post_process_AST_Retrieval(benchmark, alpha=1/3, beta=0.5, max_top_k=20, option=''):
    ###################################################################################
    # Post-process the results of abstract-query-based retrieval using AST sequence 
    # 1. Truncate by max_top_k=20
    # 2. Use mixed scores of Unfinished_only, RepoCoder, and Abstract_query_based_RACG
    ###################################################################################
        
    for repo in tqdm.tqdm(repos):
        none_retrieval = load_retrieval(f"cache/retrieval/{benchmark}/r-g/{repo}_ws20.{repo}_ws20_slice2.one-gram.top0.pkl")
        repocoder_retrieval = load_retrieval(f"cache/retrieval/{benchmark}/r-g-r-g/rg-one-gram-ws-20-ss-2_samples.{repo}_ws20.{repo}_ws20_slice2.one-gram.top0.pkl")
        abs_retrieval = load_retrieval(f"cache/abs_retrieval/{benchmark}/r-g-r-g/rg-one-gram-ws-20-ss-2_samples.{repo}_ws20{option}.{repo}_ws20_slice2.ast-sequence.top0.pkl")
        
        none_retrieval = sorted(none_retrieval, key=lambda x: x['metadata']['task_id'])
        repocoder_retrieval = sorted(repocoder_retrieval, key=lambda x: x['metadata']['task_id'])
        abs_retrieval = sorted(abs_retrieval, key=lambda x: x['metadata']['task_id'])
        
        abs_to_code_retrieval = copy.deepcopy(abs_retrieval)
        
        for i in range(len(abs_to_code_retrieval)):
            tmp_dict = {}

        #     contexts = none_retrieval[i]['top_k_context']  
            def update_dict(tmp_dict, contexts, _key):        
                for j in range(len(contexts)):
                    key = get_cand_id(contexts[j])
                    if key not in tmp_dict.keys():
                        tmp_dict[key] = {'data': None, 'none': 0., 'repocoder': 0., 'abs': 0.}

                    if _key != 'abs' and tmp_dict[key]['data'] is None:            
                        tmp_dict[key]['data'] = contexts[j][0]
                    tmp_dict[key][_key] = contexts[j][1]

                return tmp_dict

            tmp_dict = update_dict(tmp_dict, none_retrieval[i]['top_k_context'], _key='none')
            tmp_dict = update_dict(tmp_dict, repocoder_retrieval[i]['top_k_context'], _key='repocoder')
            tmp_dict = update_dict(tmp_dict, abs_retrieval[i]['top_k_context'], _key='abs')

            for j in range(len(abs_retrieval[i]['top_k_context'])):
                key = get_cand_id(abs_retrieval[i]['top_k_context'][j])
                retrieval_score = alpha*tmp_dict[key]['none'] + (1-alpha)*(beta*tmp_dict[key]['repocoder'] + (1-beta)*tmp_dict[key]['abs'])
                abs_to_code_retrieval[i]['top_k_context'][j] = (tmp_dict[key]['data'], retrieval_score)
            ########################################
            # Sort and truncate by max_top_k=20
            ########################################
            abs_to_code_retrieval[i]['top_k_context'] = sorted(abs_to_code_retrieval[i]['top_k_context'], key=lambda x: x[1])[-max_top_k:]

        Tools.dump_pickle(abs_to_code_retrieval, 
                         f"cache/abs_retrieval/{benchmark}/r-g-r-g/rg-one-gram-ws-20-ss-2_samples.{repo}_ws20.{repo}_ws20_slice2.ast-sequence_alpha={alpha}_beta={beta}{option}.top{max_top_k}.pkl")
    
#     return abs_to_code_retrieval



class BuildAbstractPromptWrapper(BuildPromptWrapper):
    ######################################################################################
    # Build the abstract prompt by abstract-query-based retrieval using AST sequence
    ######################################################################################
    def __init__(self, vectorizer, benchmark, repos, window_size, slice_size, tokenizer, full_model_name):
        super().__init__(vectorizer, benchmark, repos, window_size, slice_size, tokenizer, full_model_name)
        if vectorizer[:len('ast-sequence')] == 'ast-sequence':
            self.vector_path_builder = FilePathBuilder.ast_sequence_vector_path

    def _run(self, mode, alpha, beta, output_file_path, option):
        workers = []
        for repo in self.repos:
            retrieval_results = f"cache/abs_retrieval/{self.benchmark}/r-g-r-g/rg-one-gram-ws-20-ss-2_samples.{repo}_ws20.{repo}_ws20_slice2.ast-sequence_alpha={alpha}_beta={beta}{option}.top{self.max_top_k}.pkl"
            
            query_lines_with_retrieval_results = Tools.load_pickle(retrieval_results)
            log_message = f'repo: {repo}, window: {self.window_size}, slice: {self.slice_size}'
#             print(retrieval_results)
#             print(log_message)
            worker = PromptBuilder(query_lines_with_retrieval_results, self.task_path, 
                                   log_message, self.tokenizer, self.full_model_name)
            workers.append(worker)
        lines = []
        for worker in workers:
            lines += worker.build_2nd_stage_input_file(mode)
        
        FilePathBuilder.make_needed_dir(output_file_path)
        Tools.dump_jsonl(lines, output_file_path)
            
    def build_abstract_prediction_prompt(self, mode, alpha, beta, output_path, option=''):
        self._run(mode, alpha, beta, output_path, option)
        
        
        
if __name__ == '__main__':
    repos = ['huggingface_diffusers',
             'nerfstudio-project_nerfstudio',
             'awslabs_fortuna',
             'huggingface_evaluate',
             'google_vizier',
             'alibaba_FederatedScope',
             'pytorch_rl',
             'opendilab_ACE']

    window_sizes = [20]
    slice_sizes = [2]

    mode = CONSTANTS.rgrg
    vectorizer = ASTSequence

    
    full_model_names = ['deepseek-ai/deepseek-coder-6.7b-instruct', 'deepseek-ai/deepseek-coder-6.7b-base', 'Salesforce/codegen-6B-mono', 'Salesforce/codegen-2B-mono']


    #####################################################################################################
    # alpha * Unfinished_only + (1-alpha) * ( beta*RepoCoder + (1-beta)*Abstract_query_based_RACG )
    #####################################################################################################
    
    alpha_beta_list = [(1/3, 0.5), (0.5, 0.0), (0.0, 0.5), (0.0, 0.0)]
    
    
    #####################################################################################################
    # Include Unfinished Code Lines into the abstraction process
    #####################################################################################################
    _option = 'include_unfinished'
    
    
    option = '_' + _option

    for full_model_name in full_model_names:
        model_name = Tools.get_model_name(full_model_name)
        
        benchmarks = [CONSTANTS.api_benchmark, CONSTANTS.line_benchmark]
        if model_name == 'gpt-3.5-turbo':
            tokenizer = CodexTokenizer
        elif model_name.split('-')[0] == 'codegen':
            tokenizer = CodeGenTokenizer
            # Max token length for CodeGen is 2048
            benchmarks = [CONSTANTS.short_api_benchmark, CONSTANTS.short_line_benchmark]
        else:
            tokenizer = AutoTokenizer
            
        for benchmark in benchmarks:
            prediction_path = f'predictions/{benchmark}/{model_name}/rg-one-gram-ws-20-ss-2_samples.0.jsonl'

    
            make_abs_repo_window(repos, window_sizes, slice_sizes)

            MakeAbstractWindowWrapper(benchmark, repos, window_sizes, slice_sizes).window_for_abstract_prediction(mode, prediction_path, option=option)

            BuildAbstractVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_abstract_repo_windows()
            BuildAbstractVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_abstract_prediction_windows(mode, prediction_path, option)

            CodeSearchASTWrapper('ast-sequence', benchmark, repos, window_sizes, slice_sizes).search_ast_prediction(mode, prediction_path, option)
            CodeSearchWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes).search_prediction(mode, prediction_path)
            CodeSearchWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes).search_baseline_and_ground()

    
            for alpha, beta in alpha_beta_list:
                post_process_AST_Retrieval(benchmark, alpha=alpha, beta=beta, option=option)

                output_file_path = f'prompts/{benchmark}/{model_name}/ast-sequence_alpha={alpha}_beta={beta}{option}-ES-ws-20-ss-2.jsonl'
                BuildAbstractPromptWrapper(f'ast-sequence_alpha={alpha}_beta={beta}', benchmark, repos, window_sizes, slice_sizes, tokenizer, full_model_name).build_abstract_prediction_prompt(mode, alpha, beta, output_file_path, option)
