# model_name=deepseek-ai/deepseek-coder-6.7b-instruct
# model_name=Salesforce/codegen-2B-mono
init_device_id=0

# +
# for retrieved in rg none gt; do
#     for benchmark in random_api random_line; do
#         for ((device = $init_device_id; device < 8; device++)); do
#             python inference.py --device $device --benchmark $benchmark --retrieved $retrieved --model_name $model_name --init_device_id $init_device_id &
#         done
#         wait
#     done
# done 
# -

for model_name in Salesforce/codegen-6B-mono; do
    device=$init_device_id
    for retrieved in none; do # none rg repocoder gt oracle
        for benchmark in short_api short_line; do
            python inference.py --device $device --benchmark $benchmark --retrieved $retrieved --model_name $model_name &
            device=$((device+1))
        done
    done 
    wait
done



