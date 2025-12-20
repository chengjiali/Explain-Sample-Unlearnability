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
    # "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
)
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)



per_device_train_batch_size=4 # on two gpus would make effective batch size 32
gradient_accumulation_steps=2


# for split in "${splits[@]}"; do
#     forget_split=$(echo $split | cut -d' ' -f1)
#     holdout_split=$(echo $split | cut -d' ' -f2)
#     retain_split=$(echo $split | cut -d' ' -f3)

#     for model in "${models[@]}"; do
#         for trainer in "${trainers[@]}"; do
#             for seed in 42 87; do
#                 model_path=open-unlearning/tofu_${model}_full

#                 TRAIN_CMD="CUDA_VISIBLE_DEVICES=0,2,3,4 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
#                 src/train.py --config-name=unlearn.yaml \
#                 experiment=unlearn/tofu/default.yaml \
#                 trainer=${trainer} \
#                 model=${model} \
#                 forget_split=${forget_split} \
#                 retain_split=${retain_split} \
#                 model.model_args.pretrained_model_name_or_path=${model_path} \
#                 retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
#                 trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
#                 trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
#                 trainer.args.ddp_find_unused_parameters=true \
#                 trainer.args.gradient_checkpointing=true \
#                 trainer.args.eval_strategy=no"

#                 # Standard
#                 task_name=tofu_${model}_${forget_split}_${trainer}_standard_${seed}
#                 if [ ! -f saves/unlearn/"${task_name}"/model.safetensors ] && [ ! -f saves/unlearn/"${task_name}"/model.safetensors.index.json ]; then
#                     echo "${task_name}" "Model Not Found"

#                     eval ${TRAIN_CMD} \
#                     task_name=${task_name} \
#                     trainer.cl.method="none"
#                 fi
                
#                 # CL
#                 task_name=tofu_${model}_${forget_split}_${trainer}_cl_${seed}
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


for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        model_path=open-unlearning/tofu_${model}_full

        for mc_config in 'parameter'; do
            for curve in 'linear' ; do
                for mc_type in 'random' 'random_in_cl' 'random_in_so' 'cl-non_cl' 'fo-so'; do
                    for trainer in "${trainers[@]}"; do

                        task_name=${mc_config}/${mc_type}/${curve}/tofu_${model}_${forget_split}_${trainer}

                        EVAL_CMD="CUDA_VISIBLE_DEVICES=0 python src/eval_curve.py \
                        experiment=eval/tofu/default.yaml \
                        model=${model} \
                        forget_split=${forget_split} \
                        holdout_split=${holdout_split} \
                        model.model_args.pretrained_model_name_or_path=${model_path} \
                        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

                        if [ ! -f saves/eval_curve/"${task_name}"/eval_curve.csv ]; then
                            echo "${task_name}" "Eval Not Found"
                            
                            eval ${EVAL_CMD} \
                            task_name=${task_name} \
                            paths.output_dir=saves/eval_curve/${task_name}/
                        fi
                    done
                done

                for mc_type in 'method_fo-so'; do
                    for trainer1 in "${trainers[@]}"; do
                        for trainer2 in "${trainers[@]}"; do

                            if [[ "$trainer1" == "$trainer2" ]]; then
                                continue
                            fi

                            task_name=${mc_config}/${mc_type}/${curve}/tofu_${model}_${forget_split}_${trainer1}_${trainer2}

                            EVAL_CMD="CUDA_VISIBLE_DEVICES=0 python src/eval_curve.py \
                            experiment=eval/tofu/default.yaml \
                            model=${model} \
                            forget_split=${forget_split} \
                            holdout_split=${holdout_split} \
                            model.model_args.pretrained_model_name_or_path=${model_path} \
                            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

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
