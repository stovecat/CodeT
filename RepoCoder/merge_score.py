import os
import numpy as np
from utils import Tools

def get_n_chunk(base_path, benchmark, model, option):
    prefix = f"{option}_{model}"
    chunk_indices = []
    for fn in os.listdir(f"{base_path}/{benchmark}"):
        if fn[-4:] != ".pkl":
            continue
        if prefix == fn[:len(prefix)]:
            chunk_indices.append(int(fn[len(prefix):-4].split("_")[1]))
    assert sorted(chunk_indices) == list(range(len(chunk_indices)))
    return len(chunk_indices)


def get_scores(base_path="results", 
               benchmarks=["random_line", "random_api"],
               model="codegen-2B-mono",
               options=["none", "rg", "repocoder", 
#                         "extractive_summary",
                        "extractive_summary_identifier", 
                        "extractive_summary_entity",
                        "extractive_summary_non_entity",
                        "extractive_summary_non_entity_no_built_in_ids",
                        "gt", "oracle"],
               new_score=False):
    result_dict = {}
    for benchmark in benchmarks:
        result_dict[benchmark] = {}
        for option in options:
            result_dict[benchmark][option] = None
            n_chunk = get_n_chunk(base_path, benchmark, model, option)
            for chunk_idx in range(n_chunk):
                tmp_path = f"{base_path}/{benchmark}/{option}_{model}_{chunk_idx}.pkl"
                tmp_result = Tools.load_pickle(tmp_path)
                if result_dict[benchmark][option] is None:
                    result_dict[benchmark][option] = tmp_result
                else:
                    for key, val in tmp_result.items():
                        result_dict[benchmark][option][key].extend(val)    
                        
    def display(new_score=False):                        
        print(f"{model}")
        for benchmark in benchmarks:
            print(f"[{benchmark}]")
            print("Option\tEM\tES")
            for option in options:
                if result_dict[benchmark][option] is None:
                    print(f"{option}\t-\t-")
                    continue
                if new_score:
#                     from compute_score import compute_new_EM, compute_new_ES
                    new_EM_scores = []
                    new_ES_scores = []
                    for i in range(len(result_dict[benchmark][option]['EM_scores'])):
#                         print(result_dict[benchmark][option].keys())
                        target = result_dict[benchmark][option]['ground_truths'][i]
                        preds = result_dict[benchmark][option]['predictions'][i]
                        new_EM_scores.append(compute_new_EM(target=target, predictions=preds, passk=1))
                        new_ES_scores.append(compute_new_ES(target=target, predictions=preds, passk=1))
                    print(f"{option}\t{np.mean(new_EM_scores)}\t{np.mean(new_ES_scores)}")
                    result_dict[benchmark][option]['new_EM_scores'] = new_EM_scores
                    result_dict[benchmark][option]['new_ES_scores'] = new_ES_scores
                else:
                    print(f"{option}\t{np.mean(result_dict[benchmark][option]['EM_scores'])}\t{np.mean(result_dict[benchmark][option]['ES_scores'])}")
            print()
            
    display(new_score)
    
    return result_dict


if __name__ == '__main__':
    result_dict = get_scores()
