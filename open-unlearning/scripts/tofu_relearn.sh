#!/bin/bash


# set -e
export TRANSFORMERS_CACHE='/data_w/jiali/huggingface_cache'
export HYDRA_FULL_ERROR=1
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
)
trainers=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "SimNPO"
    # "RMU"
)
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)


per_device_train_batch_size=8 # on two gpus would make effective batch size 32
gradient_accumulation_steps=2


########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################



for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for trainer in "${trainers[@]}"; do
            
            task_name=tofu_${model}_${forget_split}_${trainer}
            model_path=open-unlearning/tofu_${model}_full

            out_dir=seed_0

            if [ ! -f saves/relearn_"${out_dir}"/"${task_name}"/adapter_model.safetensors ]; then
                echo "${task_name}" "Model Not Found"
                cd jog_llm_memory/wmdp/src
                
                # Unlearn
                # CUDA_VISIBLE_DEVICES=3,4 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
                CUDA_VISIBLE_DEVICES=0 python relearn_full.py --config-name=relearn.yaml \
                model_path=${model_path} \
                ckpt=../../../saves/unlearn_"${out_dir}"/"${task_name}" \
                save_dir=../../../saves/relearn_"${out_dir}"/"${task_name}"
                
                cd ../../../
            fi


            for step in "5" "10" "15" "20" "25"; do
                if [ ! -f saves/relearn_"${out_dir}"/"${task_name}"/checkpoint-"${step}"/evals/TOFU_SUMMARY.json ]; then
                    echo "${task_name}" "Eval Not Found"

                    # Eval
                    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                    experiment=eval/tofu/default.yaml \
                    forget_split=${forget_split} \
                    holdout_split=${holdout_split} \
                    model=${model} \
                    task_name=${task_name} \
                    model.ckpt=saves/relearn_"${out_dir}"/${task_name}/checkpoint-"${step}" \
                    model.model_args.pretrained_model_name_or_path=${model_path} \
                    paths.output_dir=saves/relearn_"${out_dir}"/${task_name}/checkpoint-"${step}"/evals \
                    retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
                fi

                # # Jailbreak 1: system prompt
                # if [ ! -f saves/unlearn_"${out_dir}"/"${task_name}"/checkpoint-"${step}"/evals_sysprompt/TOFU_SUMMARY.json ]; then
                #     echo "${task_name}"/checkpoint-"${step}" "Eval Not Found"

                #     # Eval
                #     CUDA_VISIBLE_DEVICES=0 python src/eval_sysprompt.py \
                #     experiment=eval/tofu/default.yaml \
                #     forget_split=${forget_split} \
                #     holdout_split=${holdout_split} \
                #     model=${model} \
                #     task_name=${task_name} \
                #     model.ckpt=saves/unlearn_"${out_dir}"/${task_name}/checkpoint-"${step}" \
                #     model.model_args.pretrained_model_name_or_path=${model_path} \
                #     paths.output_dir=saves/unlearn_"${out_dir}"/${task_name}/checkpoint-"${step}"/evals_sysprompt \
                #     retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
                # fi

                # # Jailbreak 2: grandmother
                # if [ ! -f saves/unlearn_"${out_dir}"/"${task_name}"/checkpoint-"${step}"/evals_grandma/TOFU_SUMMARY.json ]; then
                #     echo "${task_name}"/checkpoint-"${step}" "Eval Not Found"

                #     # Eval
                #     CUDA_VISIBLE_DEVICES=0 python src/eval_grandma.py \
                #     experiment=eval/tofu/default.yaml \
                #     forget_split=${forget_split} \
                #     holdout_split=${holdout_split} \
                #     model=${model} \
                #     task_name=${task_name} \
                #     model.ckpt=saves/unlearn_"${out_dir}"/${task_name}/checkpoint-"${step}" \
                #     model.model_args.pretrained_model_name_or_path=${model_path} \
                #     paths.output_dir=saves/unlearn_"${out_dir}"/${task_name}/checkpoint-"${step}"/evals_grandma \
                #     retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
                # fi
            done
        done
    done
done