#!/bin/bash


export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

trainers=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "SimNPO"
    "RMU"
    "WGA"
)
models=(
    zephyr-7b-beta
)
data_splits=(
    "cyber"
    "bio"
)


per_device_train_batch_size=4 # on two gpus would make effective batch size 32
gradient_accumulation_steps=2


# for data_split in "${data_splits[@]}"; do
#     for model in "${models[@]}"; do
#         for trainer in "${trainers[@]}"; do
#             for seed in 42 87; do

#                 TRAIN_CMD="CUDA_VISIBLE_DEVICES=0,2,3,4 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
#                 src/train.py --config-name=unlearn.yaml \
#                 experiment=unlearn/wmdp/default.yaml \
#                 trainer=${trainer} \
#                 model=${model} \
#                 data_split=${data_split} \
#                 trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
#                 trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
#                 trainer.args.ddp_find_unused_parameters=true \
#                 trainer.args.gradient_checkpointing=true \
#                 trainer.args.seed=${seed} \
#                 trainer.args.eval_strategy=no"

#                 EVAL_CMD="CUDA_VISIBLE_DEVICES=4 python src/eval.py \
#                 experiment=eval/wmdp/default.yaml \
#                 model=${model} \
#                 data_split=${data_split}"

#                 # Standard
#                 task_name=wmdp_${model}_${data_split}_${trainer}_standard_${seed}

#                 if [ ! -f saves/unlearn/"${task_name}"/model.safetensors ] && [ ! -f saves/unlearn/"${task_name}"/model.safetensors.index.json ]; then
#                     echo "${task_name}" "Model Not Found"

#                     eval ${TRAIN_CMD} \
#                     task_name=${task_name} \
#                     trainer.cl.method="none"
#                 fi

#                 # CL
#                 task_name=wmdp_${model}_${data_split}_${trainer}_cl_${seed}
#                 if [ ! -f saves/unlearn/"${task_name}"/model.safetensors ] && [ ! -f saves/unlearn/"${task_name}"/model.safetensors.index.json ]; then
#                     echo "${task_name}" "Model Not Found"
                    
#                     eval ${TRAIN_CMD} \
#                     task_name=${task_name} \
#                     trainer.cl.method=per_token_superloss \
#                     trainer.cl.lam=1 \
#                     trainer.cl.C=2
#                 fi

#             done
#         done
#     done
# done


for data_split in "${data_splits[@]}"; do
    for model in "${models[@]}"; do
        for mc_config in 'parameter'; do
            for curve in 'linear'; do
                for mc_type in 'random' 'random_in_cl' 'random_in_so' 'cl_non-cl' 'fo-so'; do
                    for trainer in "${trainers[@]}"; do

                        task_name=${mc_config}/${mc_type}/${curve}/wmdp_${model}_${data_split}_${trainer}

                        EVAL_CMD="CUDA_VISIBLE_DEVICES=3 python src/eval_curve.py \
                        experiment=eval/wmdp/default.yaml \
                        model=${model} \
                        data_split=${data_split}"

                        if [ ! -f saves/eval_curve/"${task_name}"/eval_curve.csv ]; then
                            echo "${task_name}" "Eval Not Found"
                            
                            eval ${EVAL_CMD} \
                            task_name=${task_name} \
                            paths.output_dir=saves/eval_curve/${task_name}/
                        fi
                    done
                done

                for mc_type in 'method' 'method_in_cl' 'method_in_so' 'method_fo-so'; do
                    for trainer1 in "${trainers[@]}"; do
                        for trainer2 in "${trainers[@]}"; do

                            if [[ "$trainer1" == "$trainer2" ]]; then
                                continue
                            fi

                            task_name=${mc_config}/${mc_type}/${curve}/wmdp_${model}_${data_split}_${trainer1}_${trainer2}

                            EVAL_CMD="CUDA_VISIBLE_DEVICES=3 python src/eval_curve.py \
                            experiment=eval/wmdp/default.yaml \
                            model=${model} \
                            data_split=${data_split}"

                            if [ ! -f saves/eval_curve/"${task_name}"/eval_curve.csv ]; then
                                echo "${task_name}" "Eval Not Found"
                                
                                eval ${EVAL_CMD} \
                                task_name=${task_name} \
                                paths.output_dir=saves/eval_curve/${task_name}/
                            fi
                        done
                    done
                done
            done
        done
    done
done