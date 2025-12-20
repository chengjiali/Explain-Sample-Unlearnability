#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


per_device_train_batch_size=4
gradient_accumulation_steps=4

out_dir=unlearn_seed_42

model=Llama-2-7b-hf

data_splits=(
    "News"
    "Books"
)

trainers=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "SimNPO"
    "RMU"
    "UNDIAL"
    "SatImp"
    "WGA"
    "CEU"
)

# #########################################################
# #################### MUSE Unlearning ####################
# #########################################################


for data_split in "${data_splits[@]}"; do
    for trainer in "${trainers[@]}"; do
        task_name=muse_${model}_${data_split}_${trainer}

        if [ ! -f saves/"${out_dir}"/"${task_name}"/adapter_model.safetensors ]; then
            echo "${task_name}" "Model Not Found"
            CUDA_VISIBLE_DEVICES=3,4 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
            src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/default.yaml \
            model=${model} \
            data_split=${data_split} \
            trainer=${trainer} \
            task_name=${task_name} \
            retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
            paths.output_dir=saves/${out_dir}/${task_name} \
            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
            trainer.args.gradient_checkpointing=true \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.eval_strategy=no \
            trainer.args.num_train_epochs=20 \
            trainer.args.seed=42
        fi

        # if [ ! -f saves/"${out_dir}"/"${task_name}"/evals/MUSE_SUMMARY.json ]; then
        #     echo saves/"${out_dir}"/"${task_name}"/evals/MUSE_SUMMARY.json "Not Found"

        #     CUDA_VISIBLE_DEVICES=3 python src/eval.py \
        #     experiment=eval/muse/default.yaml \
        #     data_split=${data_split} \
        #     task_name=${task_name} \
        #     model=${model} \
        #     model.ckpt=saves/${out_dir}/${task_name} \
        #     paths.output_dir=saves/${out_dir}/${task_name}/evals \
        #     retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
        # fi

        # # Jailbreak #1
        # if [ ! -f saves/"${out_dir}"/"${task_name}"/evals_sysprompt/MUSE_SUMMARY.json ]; then
        #     echo saves/"${out_dir}"/"${task_name}"/evals_sysprompt/MUSE_SUMMARY.json "Not Found"

        #     CUDA_VISIBLE_DEVICES=3 python src/eval_sysprompt.py \
        #     experiment=eval/muse/default.yaml \
        #     data_split=${data_split} \
        #     task_name=${task_name} \
        #     model=${model} \
        #     model.ckpt=saves/${out_dir}/${task_name} \
        #     paths.output_dir=saves/${out_dir}/${task_name}/evals_sysprompt \
        #     retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
        # fi

        # # Jailbreak #2
        # if [ ! -f saves/"${out_dir}"/"${task_name}"/evals_grandma/MUSE_SUMMARY.json ]; then
        #     echo saves/"${out_dir}"/"${task_name}"/evals_grandma/MUSE_SUMMARY.json "Not Found"

        #     CUDA_VISIBLE_DEVICES=3 python src/eval_grandma.py \
        #     experiment=eval/muse/default.yaml \
        #     data_split=${data_split} \
        #     task_name=${task_name} \
        #     model=${model} \
        #     model.ckpt=saves/${out_dir}/${task_name} \
        #     paths.output_dir=saves/${out_dir}/${task_name}/evals_grandma \
        #     retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
        # fi
    done
done



# #########################################################
# ########### MUSE News Unlearning Scalability ############
# #########################################################


# for data_split in "${data_splits[@]}"; do
#     for trainer in "${trainers[@]}"; do
#         for scal in "forget_1" "forget_2" "forget_3" "forget_4"; do
            
#             task_name=muse_${model}_${data_split}_${trainer}_scal_${scal} \
            
#             if [ ! -f saves/"${out_dir}"/"${task_name}"/adapter_model.safetensors ]; then
#                 echo "${task_name}" "Model Not Found"
#                 # CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
#                 CUDA_VISIBLE_DEVICES=3 python \
#                 src/train.py --config-name=unlearn.yaml \
#                 experiment=unlearn/muse/scalability.yaml \
#                 model=${model} \
#                 data_split=${data_split} \
#                 forget_split=${scal} \
#                 trainer=${trainer} \
#                 task_name=${task_name} \
#                 retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
#                 paths.output_dir=saves/${out_dir}/${task_name} \
#                 trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
#                 trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
#                 trainer.args.gradient_checkpointing=true \
#                 trainer.args.ddp_find_unused_parameters=true trainer.args.eval_strategy=no trainer.args.seed=0
#             fi

#             # if [ ! -f saves/"${out_dir}"/"${task_name}"/evals/MUSE_SUMMARY.json ]; then
#             #     echo saves/${out_dir}/${task_name}/evals/MUSE_SUMMARY.json "Not Found"
#             #     CUDA_VISIBLE_DEVICES=3 python src/eval.py \
#             #     experiment=eval/muse/default.yaml \
#             #     data_split=${data_split} \
#             #     task_name=${task_name} \
#             #     model=${model} \
#             #     model.ckpt=saves/${out_dir}/${task_name} \
#             #     paths.output_dir=saves/${out_dir}/${task_name}/evals \
#             #     retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
#             # fi
#         done
#     done
# done



#########################################################
########### MUSE News Unlearning sustainability #########
#########################################################


# for data_split in "${data_splits[@]}"; do
#     for trainer in "${trainers[@]}"; do
#         model_path=muse-bench/MUSE-${data_split}_target
#         for sust in "forget_1" "forget_2" "forget_3" "forget_4"; do
            
#             if [ ! -f saves/"${out_dir}"/"${task_name}"/adapter_model.safetensors ]; then
#                 echo "${task_name}" "Model Not Found"
#                 task_name=muse_${model}_${data_split}_${trainer}_sust_${sust}

#                 # CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
#                 CUDA_VISIBLE_DEVICES=3 python \
#                 src/train.py --config-name=unlearn.yaml \
#                 experiment=unlearn/muse/sustainabilty.yaml \
#                 model=${model} \
#                 model.model_args.pretrained_model_name_or_path=${model_path} \
#                 data_split=${data_split} \
#                 trainer=${trainer} \
#                 task_name=${task_name} \
#                 retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
#                 trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
#                 trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                # trainer.args.gradient_checkpointing=true \
#                 trainer.args.ddp_find_unused_parameters=true trainer.args.eval_strategy=no trainer.args.seed=0 \
#             fi

            # if [ ! -f saves/"${out_dir}"/"${task_name}"/evals/MUSE_SUMMARY.json ]; then
            #     echo "${task_name}" "Eval Not Found"
#             # CUDA_VISIBLE_DEVICES=3 python src/eval.py \
#             # experiment=eval/muse/default.yaml \
#             # data_split=${data_split} \
#             # task_name=${task_name} \
#             # model=${model} \
#             # model.ckpt=saves/"${out_dir}"/${task_name} \
#             # paths.output_dir=saves/"${out_dir}"/${task_name}/evals \
#             # retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json

#             ckpt=saves/"${out_dir}"/${task_name}
            # fi
#         done
#     done
# done