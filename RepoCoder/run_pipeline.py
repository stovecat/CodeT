# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from make_window import MakeWindowWrapper
from build_vector import BuildVectorWrapper, BagOfWords
from search_code import CodeSearchWrapper
from build_prompt import BuildPromptWrapper

from utils import CONSTANTS, CodexTokenizer, CodeGenTokenizer, AutoTokenizer, Tools

def make_repo_window(repos, window_sizes, slice_sizes):
    worker = MakeWindowWrapper(None, repos, window_sizes, slice_sizes)
    worker.window_for_repo_files()


def run_RG1_and_oracle_method(benchmark, repos, window_sizes, slice_sizes, tokenizer, full_model_name):
    # build code snippets for all the repositories
    make_repo_window(repos, window_sizes, slice_sizes)
    # build code snippets for vanilla retrieval-augmented approach and ground truth
    MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes).window_for_baseline_and_ground()
    # build vector for vanilla retrieval-augmented approach and ground truth
    vectorizer = BagOfWords
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_baseline_and_ground_windows()
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_repo_windows()
    # search code for vanilla retrieval-augmented approach and ground truth
    CodeSearchWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes).search_baseline_and_ground()

    # build prompt for vanilla retrieval-augmented approach and ground truth
    model_name = Tools.get_model_name(full_model_name)
    mode = CONSTANTS.rg
    output_file_path = f'prompts/{benchmark}/{model_name}/rg-one-gram-ws-20-ss-2.jsonl'
    BuildPromptWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes, tokenizer, full_model_name).build_first_search_prompt(mode, output_file_path)

    mode = CONSTANTS.gt
    output_file_path = f'prompts/{benchmark}/{model_name}/gt-one-gram-ws-20-ss-2.jsonl'
    BuildPromptWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes, tokenizer, full_model_name).build_first_search_prompt(mode, output_file_path)


def run_RepoCoder_method(benchmark, repos, window_sizes, slice_sizes, prediction_path, tokenizer, full_model_name):
    model_name = Tools.get_model_name(full_model_name)
    mode = CONSTANTS.rgrg
    MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes).window_for_prediction(mode, prediction_path)
    vectorizer = BagOfWords
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_prediction_windows(mode, prediction_path)
    CodeSearchWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes).search_prediction(mode, prediction_path)
    output_file_path = f'prompts/{benchmark}/{model_name}/repocoder-one-gram-ws-20-ss-2.jsonl'
    BuildPromptWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes, tokenizer, full_model_name).build_prediction_prompt(mode, prediction_path, output_file_path)


def run_custom_method(benchmark, repos, window_sizes, slice_sizes, 
                      prediction_path, tokenizer, full_model_name, build_option):
    model_name = Tools.get_model_name(full_model_name)
    mode = CONSTANTS.rgrg
    MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes).window_for_prediction(mode, prediction_path)
    vectorizer = BagOfWords
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_prediction_windows(mode, prediction_path)
    CodeSearchWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes).search_prediction(mode, prediction_path)
    output_file_path = f'prompts/{benchmark}/{model_name}/{build_option}-one-gram-ws-20-ss-2.jsonl'
    BuildPromptWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes, tokenizer, full_model_name).build_prediction_prompt(mode, prediction_path, output_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repos', type=list, default=['huggingface_diffusers',
                                                       'nerfstudio-project_nerfstudio',
                                                       'awslabs_fortuna',
                                                       'huggingface_evaluate',
                                                       'google_vizier',
                                                       'alibaba_FederatedScope',
                                                       'pytorch_rl',
                                                       'opendilab_ACE'])
    parser.add_argument('--window_sizes', type=list, default=[20])
    parser.add_argument('--slice_sizes', type=list, default=[2])
    parser.add_argument('--full_model_name', type=str, 
                        default='Salesforce/codegen-2B-mono', choices=['Salesforce/codegen-350M-mono',
                                                                       'Salesforce/codegen-2B-mono',
                                                                       'Salesforce/codegen-6B-mono',
                                                                       'deepseek-ai/deepseek-coder-6.7b-base',
                                                                       'deepseek-ai/deepseek-coder-6.7b-instruct'])
    parser.add_argument('--build_option', type=str, default="rg1_oracle", choices=["rg1_oracle", 
                                                                                   "repocoder", 
                                                                                   "extractive_summary", 
                                                                                   "abstractive_summary"])
    parser.add_argument('--prediction_fn', type=str, default='rg-one-gram-ws-20-ss-2_samples.0.jsonl')

#     repos = [
#         'huggingface_diffusers',
#         'nerfstudio-project_nerfstudio',
#         'awslabs_fortuna',
#         'huggingface_evaluate',
#         'google_vizier',
#         'alibaba_FederatedScope',
#         'pytorch_rl',
#         'opendilab_ACE',
#     ]
    
#     window_sizes = [20]
#     slice_sizes = [2]  # 20 / 2 = 10

#     full_model_name = 'Salesforce/codegen-6B-mono' # 'deepseek-ai/deepseek-coder-6.7b-instruct' # 

    args = parser.parse_args()
    model_name = Tools.get_model_name(args.full_model_name)
    
    benchmarks = [CONSTANTS.api_benchmark, CONSTANTS.line_benchmark]
    if model_name == 'gpt-3.5-turbo':
        tokenizer = CodexTokenizer
    elif model_name.split('-')[0] == 'codegen':
        tokenizer = CodeGenTokenizer
        # Max token length for CodeGen is 2048
        benchmarks = [CONSTANTS.short_api_benchmark, CONSTANTS.short_line_benchmark]
    else:
        tokenizer = AutoTokenizer
        
        
    if args.build_option == 'rg1_oracle':
        # build prompt for the RG1 and oracle methods    
        print("build prompt for the RG1 and oracle methods")
        for benchmark in benchmarks:
            run_RG1_and_oracle_method(benchmark, args.repos, args.window_sizes, 
                                      args.slice_sizes, tokenizer, args.full_model_name)
    elif args.build_option == 'repocoder':
        # build prompt for the RepoCoder method
        print("build prompt for the RepoCoder method")
        for benchmark in benchmarks:
            prediction_path = f'predictions/{benchmark}/{model_name}/{args.prediction_fn}'
            if os.path.exists(prediction_path):
                run_RepoCoder_method(benchmark, args.repos, args.window_sizes, args.slice_sizes, 
                                     prediction_path, tokenizer, args.full_model_name)
            else:
                print(f"[ERROR] FILE NOT EXIST: {prediction_path}")
    elif args.build_option in ["extractive_summary", "abstractive_summary"]:
        # build prompt for the custom method
        print(f"build prompt for the {args.build_option} method")
        for benchmark in benchmarks:
            prediction_path = f'predictions/{benchmark}/{model_name}/{args.prediction_fn}'
            if os.path.exists(prediction_path):
                run_custom_method(benchmark, args.repos, args.window_sizes, args.slice_sizes, 
                                  prediction_path, tokenizer, args.full_model_name, args.build_option)
            else:
                print(f"[ERROR] FILE NOT EXIST: {prediction_path}")


