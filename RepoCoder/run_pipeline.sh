# +
# full_model_name=['Salesforce/codegen-350M-mono',
#                  'Salesforce/codegen-2B-mono',
#                  'Salesforce/codegen-6B-mono',
#                  'deepseek-ai/deepseek-coder-6.7b-base'
#                  'deepseek-ai/deepseek-coder-6.7b-instruct']
full_model_name=$1

# build_option: ["rg1_oracle", "repocoder", "extractive_summary", "abstractive_summary"]
build_option=$2 

# For Iterative RAG methods (i.e. RGRG)
# prediction_fn='rg-one-gram-ws-20-ss-2_samples.0.jsonl' # RepoCoder
postfix='-one-gram-ws-20-ss-2_samples.0.jsonl'
if [ $build_option == "repocoder" ]; then
    prediction_fn=rg${postfix}
elif [ $build_option == "extractive_summary" ] || \
     [ $build_option == "abstractive_summary" ] || \
     [ $build_option == "extractive_summary_omission" ] || \
     [ $build_option == "extractive_summary_identifier" ]; then
    prediction_fn=${build_option}${postfix}
else 
    prediction_fn=none
fi 
# -

python run_pipeline.py \
    --full_model_name $full_model_name \
    --build_option $build_option \
    --prediction_fn $prediction_fn 
