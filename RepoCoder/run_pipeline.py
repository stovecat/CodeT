# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from make_window import MakeWindowWrapper
from build_vector import BuildVectorWrapper, BagOfWords
from search_code import CodeSearchWrapper
from build_prompt import BuildPromptWrapper

from utils import CONSTANTS, CodexTokenizer, CodeGenTokenizer

def make_repo_window(repos, window_sizes, slice_sizes):
    worker = MakeWindowWrapper(None, repos, window_sizes, slice_sizes)
    worker.window_for_repo_files()


def run_RG1_and_oracle_method(benchmark, repos, window_sizes, slice_sizes, tokenizer):
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
    mode = CONSTANTS.rg
    output_file_path = f'prompts/{benchmark}/rg-one-gram-ws-20-ss-2.jsonl'
    BuildPromptWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes, tokenizer).build_first_search_prompt(mode, output_file_path)

    mode = CONSTANTS.gt
    output_file_path = f'prompts/{benchmark}/gt-one-gram-ws-20-ss-2.jsonl'
    BuildPromptWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes, tokenizer).build_first_search_prompt(mode, output_file_path)


def run_RepoCoder_method(benchmark, repos, window_sizes, slice_sizes, prediction_path, tokenizer):
    mode = CONSTANTS.rgrg
    MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes).window_for_prediction(mode, prediction_path)
    vectorizer = BagOfWords
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_prediction_windows(mode, prediction_path)
    CodeSearchWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes).search_prediction(mode, prediction_path)
    output_file_path = f'prompts/{benchmark}/repocoder-one-gram-ws-20-ss-2.jsonl'
    BuildPromptWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes, tokenizer).build_prediction_prompt(mode, prediction_path, output_file_path)


if __name__ == '__main__':
    repos = [
        'huggingface_diffusers',
        'nerfstudio-project_nerfstudio',
        'awslabs_fortuna',
        'huggingface_evaluate',
        'google_vizier',
        'alibaba_FederatedScope',
        'pytorch_rl',
        'opendilab_ACE',
    ]
    window_sizes = [20]
    slice_sizes = [2]  # 20 / 2 = 10
    length = "1k"
    
    if length == "2k":
        benchmarks = [CONSTANTS.api_benchmark, CONSTANTS.line_benchmark]
        tokenizer = CodexTokenizer
    else:
        benchmarks = [CONSTANTS.short_api_benchmark, CONSTANTS.short_line_benchmark]
        tokenizer = CodeGenTokenizer
        
    # build prompt for the RG1 and oracle methods    
    print("build prompt for the RG1 and oracle methods")
    for benchmark in benchmarks:
        run_RG1_and_oracle_method(benchmark, repos, window_sizes, slice_sizes, tokenizer)
    
    # build prompt for the RepoCoder method
    print("build prompt for the RepoCoder method")
    for benchmark in benchmarks:
        prediction_path = f'predictions/{benchmark}/rg-one-gram-ws-20-ss-2_samples.0.jsonl'
        if os.path.exists(prediction_path):
            run_RepoCoder_method(benchmark, repos, window_sizes, slice_sizes, prediction_path, tokenizer)


