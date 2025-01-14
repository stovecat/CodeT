import argparse
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import Tools, FilePathBuilder
from tqdm import tqdm



splitter = "# --------------------------------------------------\n"
precision = torch.bfloat16


# +
def commentize(text):
    return '\n#'.join(text.split('\n'))

def get_oracle_retrieved(data):
    _prompt = data["prompt"]
    _metadata = data["metadata"]
    context = commentize(_metadata['query_window']['context'])
    ground_truth = commentize(_metadata['ground_truth'])
    input_path = "/".join(_metadata['fpath_tuple'])

    oracle_retrieved = f"\
# Here are some relevant code fragments from other files of the repo:\n\
# --------------------------------------------------\n\
# the below code fragment can be found in:\n\
# examples/text_to_image/train_text_to_image.py\n\
# --------------------------------------------------\n\
# {context}\n{ground_truth}\n{splitter}"
    return oracle_retrieved 


# -

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=4, help='torch device')
    parser.add_argument('--benchmark', type=str, default="random_api", choices=["short_api", "short_line", "random_api", "random_line"], help='torch device')
    parser.add_argument('--retrieved', type=str, default="rg", choices=["repocoder", "rg", "gt", 
                                                                        "none", "oracle",
                                                                        "extractive_summary",
                                                                        "extractive_summary_identifier",
                                                                        'extractive_summary_entity',
                                                                        'extractive_summary_non_entity',
                                                                        'extractive_summary_non_entity_no_built_in_ids',
                                                                        'ast-sequence'],
                                                                help='retrieval option')
    parser.add_argument('--model_name', type=str, default="Salesforce/codegen-2B-mono", help='model name')
    parser.add_argument('--alpha', type=float, default=1/3, help='unfinished code ratio')
    parser.add_argument('--beta', type=float, default=0.5, help='predicted code line ratio')
    parser.add_argument('--option', type=str, default="", help='include_unfinished_code_for_abstract_query_generation')
#     parser.add_argument('--n_gpus', type=int, default=4, help='num_of_gpus')
#     parser.add_argument('--init_device_id', type=int, default=4, help='initial device id. We assume that the gpus are assigned in a series')

    args = parser.parse_args()
    model_name = args.model_name.split('/')[-1]
    
    option = args.option
    if args.option != '':
        option = '_'+args.option
    
    if model_name.split('-')[0] == 'codegen':
        max_token_len = 2048
    else:
        max_token_len = 4096
    max_new_tokens = 100
    max_input_len = max_token_len - max_new_tokens
    max_retrieved_code_len = max_input_len // 2
    max_input_code_len = max_input_len - max_retrieved_code_len
        
    print(args)
    if args.retrieved in ['none', 'oracle']:
        data_path = f"prompts/{args.benchmark}/{model_name}/rg-one-gram-ws-20-ss-2.jsonl"    
    elif args.retrieved == 'ast-sequence':
        data_path = f"prompts/{args.benchmark}/{model_name}/{args.retrieved}_alpha={args.alpha}_beta={args.beta}{option}-ES-ws-20-ss-2.jsonl" 
    else:
        data_path = f"prompts/{args.benchmark}/{model_name}/{args.retrieved}-one-gram-ws-20-ss-2.jsonl"  
    data = sorted(Tools.load_jsonl(data_path), 
                  key=lambda x: int(x["metadata"]["task_id"].split("/")[1]))

#     current_chunk_idx = args.device-args.init_device_id
#     chunk_size = len(data)//args.n_gpus
#     data = data[current_chunk_idx*chunk_size:(current_chunk_idx+1)*chunk_size]
    current_chunk_idx = 0
    
    # Load models
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    model = model.to(precision)
    model.to(args.device)
    model.eval()

    from compute_score import compute_EM, compute_ES
    
    predictions = []
    ground_truths = []
    EM_scores = []
    ES_scores = []
    
    with torch.no_grad():
        for i in tqdm(range(len(data))):
            _prompt = data[i]["prompt"]
            retrieved_code, input_code = _prompt.split(splitter+"\n")
            if args.retrieved == "oracle":
                retrieved_code = get_oracle_retrieved(data[i])
            retrieved_code += splitter

            # Truncate from rightside
            tokenizer.truncation_side='right'
            retrieved_tensor = tokenizer(retrieved_code, truncation=True, 
                                         max_length=max_retrieved_code_len, 
                                         return_tensors="pt")

            # Truncate from leftside
            tokenizer.truncation_side='left'
            input_tensor = tokenizer(input_code, truncation=True, 
                                     max_length=max_input_code_len, 
                                     return_tensors="pt")

            if args.retrieved == "none":
                rag_input_tensor = input_tensor.to(args.device)
            else:
                rag_input_tensor = {k: torch.cat([retrieved_tensor[k], 
                                                  input_tensor[k]], axis=1).to(args.device) 
                                    for k in input_tensor.keys()}

            completion = model.generate(**rag_input_tensor, max_new_tokens=max_new_tokens)

            _preds = [tokenizer.convert_ids_to_tokens(l) for l in completion.cpu().numpy()[:,-max_new_tokens:]]

            preds = [tokenizer.convert_tokens_to_string(p) for p in _preds]

            target = data[i]["metadata"]['ground_truth']

            em_score = compute_EM(target=target, predictions=preds, passk=1)
            es_score = compute_ES(target=target, predictions=preds, passk=1)

            predictions.append(preds)
            ground_truths.append(target)
            EM_scores.append(em_score)
            ES_scores.append(es_score)
    
    save_path = f"results/{args.benchmark}/{args.retrieved}_{model_name}_{current_chunk_idx}.pkl"
    if args.retrieved == 'ast-sequence':
        save_path = f"results/{args.benchmark}/{args.retrieved}_alpha={args.alpha}_beta={args.beta}{option}_{model_name}_{current_chunk_idx}.pkl"
    FilePathBuilder.make_needed_dir(save_path)
    Tools.dump_pickle({"predictions": predictions,
                       "ground_truths": ground_truths,
                       "EM_scores": EM_scores, 
                       "ES_scores": ES_scores}, save_path)


