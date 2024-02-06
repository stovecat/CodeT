# model_name=deepseek-ai/deepseek-coder-6.7b-instruct
# model_name=Salesforce/codegen-2B-mono
init_device_id=7

# +
# for retrieved in rg none gt; do
#     for benchmark in random_api random_line; do
#         for ((device = $init_device_id; device < 8; device++)); do
#             python inference.py --device $device --benchmark $benchmark --retrieved $retrieved --model_name $model_name --init_device_id $init_device_id &
#         done
#         wait
#     done
# done 

# +
alpha=0.0
beta=1.0

for model_name in deepseek-ai/deepseek-coder-6.7b-instruct; do
    device=$init_device_id
    for retrieved in ast-sequence; do
        for benchmark in random_api random_line; do
            python inference.py --alpha $alpha --beta $beta --device $device --benchmark $benchmark --retrieved $retrieved --model_name $model_name &
            device=$((device+1))
        done
    done 
    wait
done
# -



