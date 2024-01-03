model_name=Salesforce/codegen-2B-mono
n_gpus=2
init_device_id=2

for retrieved in repocoder; do
    for benchmark in random_api random_line; do
        for ((device = $init_device_id; device < `expr $init_device_id + $n_gpus` ; device++)); do
        python inference.py --device $device --benchmark $benchmark --retrieved $retrieved --model_name $model_name --n_gpus $n_gpus --init_device_id $init_device_id &
        done
        wait
    done
done 
