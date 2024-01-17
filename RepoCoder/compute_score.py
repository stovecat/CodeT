# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import editdistance
from collections import defaultdict

from utils import Tools

def compute_EM(target, predictions, passk):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    EM_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
        if len(target_lines) != len(prediction_lines):
            EM_scores.append(0)
            continue
        if target_lines == prediction_lines:
            EM_scores.append(1)
            continue
        EM_scores.append(0)
    return any(EM_scores)

def compute_ES(target, predictions, passk):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_str = '\n'.join(target_lines)
    ES_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
        prediction_str = '\n'.join(prediction_lines)
        ES_scores.append(
            1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))
        )
    return max(ES_scores)

# +
import editdistance

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
    

def compute_new_EM(target, predictions, passk):
    _target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines, _flag = remove_multiline_paranthesis(_target_lines)
#     if _flag:
#         print(f"{target_lines=}")
#         print("="*50)
        
    EM_scores = []
    for i, prediction in enumerate(predictions[:passk]):
        _prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
        prediction_lines, flag = remove_multiline_paranthesis(_prediction_lines)
        prediction_lines = prediction_lines[:len(target_lines)]
        _prediction_lines = _prediction_lines[:len(_target_lines)]

        if len(target_lines) != len(prediction_lines):
            EM_scores.append(0)
            continue
        ################################################################################
        # When target lines are truncated, truncate prediction lines as well
        ################################################################################
        if len(target_lines) == 1 and len(prediction_lines[0]) > len(target_lines[0]):
             prediction_lines = [p[:len(t)] for p, t in zip(prediction_lines, target_lines)]
        if target_lines == prediction_lines:
            EM_scores.append(1)
            continue
        EM_scores.append(0)
    return any(EM_scores)


def compute_new_ES(target, predictions, passk):
    _target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines, flag = remove_multiline_paranthesis(_target_lines)
    _target_str = '\n'.join(_target_lines)
    target_str = '\n'.join(target_lines)
    ES_scores = []
    for i, prediction in enumerate(predictions[:passk]):
        _prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
        prediction_lines, flag = remove_multiline_paranthesis(_prediction_lines)
        _prediction_lines = _prediction_lines[:len(_target_lines)]
        prediction_lines = prediction_lines[:len(target_lines)]
        _prediction_str = '\n'.join(_prediction_lines)
        ################################################################################
        # When target lines are truncated, truncate prediction lines as well
        ################################################################################
        truncated_prediction_lines = [p[:len(t)] for p, t in zip(prediction_lines, target_lines)]
        truncated_prediction_str = '\n'.join(truncated_prediction_lines)
        prediction_str = '\n'.join(prediction_lines)
        def get_es_score(t,p):
            return 1 - (editdistance.eval(t, p) / max(len(t), len(p)))
        ES_scores.append(
            max(get_es_score(target_str, prediction_str), get_es_score(target_str, truncated_prediction_str))
        )
        old_ES_score = get_es_score(_target_str, _prediction_str)

    return max(ES_scores)


# -

def compute_score_by_repo_with_metadata(repos, lines, stype, passk=1):
    scores = defaultdict(list)
    for line in lines:
        repo = line['metadata']['task_id'].split('/')[0]
        if repo not in repos:
            continue
        samples = [line['choices'][i]['text'] for i in range(len(line['choices']))]
        if stype == 'EM':
            score = compute_EM(line['metadata']['ground_truth'], samples, passk)
        elif stype == 'ES':
            score = compute_ES(line['metadata']['ground_truth'], samples, passk)
        scores[repo].append(score)
    avg_scores = {repo: round(sum(scores[repo]) / len(scores[repo]), 4) for repo in scores}
    repo_count = {repo: len(scores[repo]) for repo in scores}
    print(stype)
    for repo in avg_scores.keys():
        print(f'{avg_scores[repo]}\t{repo_count[repo]}\t{repo}')

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
    '''compute single prediction'''
    file_path = 'output/line-rgrg-ada-ws-20-ss-2_samples.0.jsonl'
    compute_score_by_repo_with_metadata(repos, Tools.load_jsonl(file_path), 'EM', passk=1)
    compute_score_by_repo_with_metadata(repos, Tools.load_jsonl(file_path), 'ES', passk=1)
