retrieved=rg
model_name=Salesforce/codegen-2B-mono
n_gpus=4
init_device_id=4


# +
benchmark=random_api

for retrieved in rg none gt; do
    for benchmark in random_api random_line; do
        for ((device = $init_device_id; device < 8; device++)); do
            python inference.py --device $device --benchmark $benchmark --retrieved $retrieved --model_name $model_name --init_device_id $init_device_id &
        done
        wait
    done
done 
