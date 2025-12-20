#!/bin/bash

export tokenizer_parallelism=true

gpuid=0
trainers=(
    # "GradAscent"
    # "WGA"
    # "Original open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
    # "GradDiff open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10"
    # "NPO open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.5_alpha1_epoch10"
    "SimNPO open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5"
    # "RMU open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer10_scoeff100_epoch5"
    # "UNDIAL open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha2_epoch10"
)


## TOFU
model="Llama-3.2-1B-Instruct"
# model="Llama-3.1-8B-Instruct"
splits=(
    # "forget01 holdout01 retain99"
    # "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

# for hardness in 'easy' 'hard'; do
#     for split in "${splits[@]}"; do
#         forget_split=$(echo $split | cut -d' ' -f1)
#         holdout_split=$(echo $split | cut -d' ' -f2)
#         retain_split=$(echo $split | cut -d' ' -f3)

#         for tr in "${trainers[@]}"; do
#             model_path=open-unlearning/tofu_${model}_full
#             trainer=$(echo $tr | cut -d' ' -f1)
#             ckpt=$(echo $tr | cut -d' ' -f2)

#             task_name=tofu_${model}_${forget_split}_${trainer}_${hardness}
#             # _standard_42
#             # task_name=none/tofu_${model}_${forget_split}_${trainer}_standard_42

#             if [ ! -f saves/circuit/original/${task_name}.jsonaa ]; then
#                 echo "${task_name} Circuit Not Found"

#                 if [ "$trainer" = "Original" ]; then
#                     CUDA_VISIBLE_DEVICES=$gpuid python src/find_circuit.py \
#                     experiment=eval/tofu/default.yaml \
#                     task_name=${task_name} \
#                     forget_split=${forget_split} \
#                     holdout_split=${holdout_split} \
#                     model=${model} \
#                     model.model_args.pretrained_model_name_or_path=${model_path}

#                 else
#                     CUDA_VISIBLE_DEVICES=$gpuid python src/find_circuit.py \
#                     experiment=eval/tofu/default.yaml \
#                     task_name=${task_name} \
#                     forget_split=${forget_split} \
#                     holdout_split=${holdout_split} \
#                     model=${model} \
#                     model.model_args.pretrained_model_name_or_path=${ckpt}
#                 fi
#             fi
#         done
#     done
# done

# Eval circuit
for hardness in 'easy' 'hard'; do
    for split in "${splits[@]}"; do
        forget_split=$(echo $split | cut -d' ' -f1)
        holdout_split=$(echo $split | cut -d' ' -f2)
        retain_split=$(echo $split | cut -d' ' -f3)

        for tr in "${trainers[@]}"; do
            model_path=open-unlearning/tofu_${model}_full
            trainer=$(echo $tr | cut -d' ' -f1)
            ckpt=$(echo $tr | cut -d' ' -f2)

            task_name=tofu_${model}_${forget_split}_${trainer}_${hardness}
            # _standard_42
            # task_name=none/tofu_${model}_${forget_split}_${trainer}_standard_42

            if [ ! -f saves/circuit/original/${task_name}_top400_lossa.csv ]; then
                echo "${task_name} Eval Not Found"

                if [ "$trainer" = "Original" ]; then
                    CUDA_VISIBLE_DEVICES=$gpuid python src/eval_circuit.py \
                    experiment=eval/tofu/default.yaml \
                    task_name=${task_name} \
                    forget_split=${forget_split} \
                    holdout_split=${holdout_split} \
                    model=${model} \
                    model.model_args.pretrained_model_name_or_path=${model_path} \
                    paths.output_dir=saves/circuit/${task_name}/evals \
                    retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

                else
                    CUDA_VISIBLE_DEVICES=$gpuid python src/eval_circuit.py \
                    experiment=eval/tofu/default.yaml \
                    task_name=${task_name} \
                    forget_split=${forget_split} \
                    holdout_split=${holdout_split} \
                    model=${model} \
                    model.model_args.pretrained_model_name_or_path=${ckpt} \
                    paths.output_dir=saves/circuit/${task_name}/evals \
                    retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
                fi
            fi
        done
    done
done
