from utils import Tools, FilePathBuilder

def get_prompt(data_path):
    return sorted(Tools.load_jsonl(data_path), 
                  key=lambda x: int(x["metadata"]["task_id"].split("/")[1]))

if __name__ == '__main__':
    benchmark = 'random_api'
    model_name = 'codegen-2B-mono'
    chunk_size = 4
    predictions = 'rg-one-gram-ws-20-ss-2'
    iteration = 0

    rg_prompts = get_prompt(f"prompts/{benchmark}/{predictions}.jsonl")
    samples = None
    for chunk_idx in range(chunk_size):
        tmp_samples = Tools.load_pickle(f"results/{benchmark}/rg_{model_name}_{chunk_idx}.pkl")        
        if samples is None:
            samples = tmp_samples
        else:
            for key, val in tmp_samples.items():
                samples[key].extend(val)
                
                
    sample_dict_list = []
    for i in range(len(rg_prompts)):
        meta_dict = {key: rg_prompts[i]['metadata'][key] 
                              for key in ["task_id", "fpath_tuple", \
                                          "line_no", "context_start_lineno"]}

        sample_dict = {'metadata': meta_dict, 
                           'choices': [{'text': v} for v in samples['predictions'][i]]}
        sample_dict_list.append(sample_dict)
        
    save_path = f'predictions/{benchmark}/{predictions}_samples.{iteration}.jsonl'
    FilePathBuilder.make_needed_dir(save_path)
    Tools.dump_jsonl(sample_dict_list, save_path)