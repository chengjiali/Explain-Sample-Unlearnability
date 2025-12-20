#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


per_device_train_batch_size=1
gradient_accumulation_steps=8


model=Llama-2-7b-hf

data_splits=(
    "News"
    "Books"
)

trainers=(
    "GradAscent"
    "GradDiff"
    # "NPO"
    "SimNPO"
)

# #########################################################
# #################### MUSE Unlearning ####################
# #########################################################


export out_dir=seed_42
for data_split in "${data_splits[@]}"; do
    for trainer in "${trainers[@]}"; do

        task_name=muse_${model}_${data_split}_${trainer}

        if [ ! -f saves/relearn_"${out_dir}"/"${task_name}"/adapter_model.safetensors ]; then
            echo "${task_name}" "Model Not Found"
            cd jog_llm_memory/wmdp/src
                
            # Unlearn
            # CUDA_VISIBLE_DEVICES=3,4 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
            CUDA_VISIBLE_DEVICES=4 python relearn_full.py --config-name=relearn.yaml \
            model_path=muse-bench/MUSE-${data_split}_target \
            ckpt=../../../saves/unlearn_${out_dir}/${task_name} \
            save_dir=../../../saves/relearn_${out_dir}/${task_name} \
            data_name=muse_${data_split,,}
            
            cd ../../../ 
        fi

        if [ "$data_split" == "News" ]; then
            steps=("12" "25" "37" "50" "62")
        elif [ "$data_split" == "Books" ]; then
            steps=("1" "2")
        else
            echo "Unknown split: $split"
            exit 1
        fi

        for step in "${steps[@]}"; do
            if [ ! -f saves/relearn_"${out_dir}"/"${task_name}"/checkpoint-"${step}"/evals/MUSE_SUMMARY.json ]; then
                echo saves/relearn_"${out_dir}"/"${task_name}"/checkpoint-"${step}"/evals/MUSE_SUMMARY.json "Not Found"

                CUDA_VISIBLE_DEVICES=4 python src/eval.py \
                experiment=eval/muse/default.yaml \
                data_split=${data_split} \
                task_name=${task_name} \
                model=${model} \
                model.ckpt=saves/relearn_${out_dir}/${task_name}/checkpoint-${step} \
                paths.output_dir=saves/relearn_${out_dir}/${task_name}/checkpoint-${step}/evals \
                retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
            fi
        done
    done
done



# #########################################################
# ########### MUSE News Unlearning Scalability ############
# #########################################################


for data_split in "${data_splits[@]}"; do
    for trainer in "${trainers[@]}"; do
        for scal in "forget_1" "forget_2" "forget_3" "forget_4"; do
            
            task_name=muse_${model}_${data_split}_${trainer}_scal_${scal} \
            
            if [ ! -f saves/relearn_"${out_dir}"/"${task_name}"/adapter_model.safetensors ]; then
                echo "${task_name}" "Model Not Found"
                cd jog_llm_memory/wmdp/src
                
                # Unlearn
                # CUDA_VISIBLE_DEVICES=3,4 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
                CUDA_VISIBLE_DEVICES=4 python relearn_full.py --config-name=relearn.yaml \
                model_path=muse-bench/MUSE-${data_split}_target \
                ckpt=../../../saves/unlearn_${out_dir}/${task_name} \
                save_dir=../../../saves/relearn_${out_dir}/${task_name} \
                data_name=muse_${data_split,,}
                
                cd ../../../ 
            fi

            if [ "$data_split" == "News" ]; then
                steps=("12" "25" "37" "50" "62")
            elif [ "$data_split" == "Books" ]; then
                steps=("1" "2")
            else
                echo "Unknown split: $split"
                exit 1
            fi

            for step in "${steps[@]}"; do
                if [ ! -f saves/relearn_"${out_dir}"/"${task_name}"/checkpoint-"${step}"/evals/MUSE_SUMMARY.json ]; then
                    echo saves/relearn_"${out_dir}"/"${task_name}"/checkpoint-"${step}"/evals/MUSE_SUMMARY.json "Not Found"
            
                    CUDA_VISIBLE_DEVICES=4 python src/eval.py \
                    experiment=eval/muse/default.yaml \
                    data_split=${data_split} \
                    task_name=${task_name} \
                    model=${model} \
                    model.ckpt=saves/relearn_${out_dir}/${task_name}/checkpoint-${step} \
                    paths.output_dir=saves/relearn_${out_dir}/${task_name}/checkpoint-${step}/evals \
                    retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
                fi
            done
        done
    done
done



#########################################################
########### MUSE News Unlearning sustainability #########
#########################################################


# for data_split in "${data_splits[@]}"; do
#     for trainer in "${trainers[@]}"; do
#         model_path=muse-bench/MUSE-${data_split}_target
#         for sust in "forget_1" "forget_2" "forget_3" "forget_4"; do
            
#             if [ ! -f saves/unlearn_"${out_dir}"/"${task_name}"/adapter_model.safetensors ]; then
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

            # if [ ! -f saves/unlearn_"${out_dir}"/"${task_name}"/evals/MUSE_SUMMARY.json ]; then
            #     echo "${task_name}" "Eval Not Found"
#             # CUDA_VISIBLE_DEVICES=3 python src/eval.py \
#             # experiment=eval/muse/default.yaml \
#             # data_split=${data_split} \
#             # task_name=${task_name} \
#             # model=${model} \
#             # model.ckpt=saves/unlearn_"${out_dir}"/${task_name} \
#             # paths.output_dir=saves/unlearn_"${out_dir}"/${task_name}/evals \
#             # retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json

#             ckpt=saves/unlearn_"${out_dir}"/${task_name}
            # fi
#         done
#     done
# done