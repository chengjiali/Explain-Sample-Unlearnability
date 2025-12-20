#!/bin/bash

export tokenizer_parallelism=true

trainers=(
    # "GradAscent"
    # "WGA"
    "Original open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
    "GradDiff open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10"
    "NPO open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.5_alpha1_epoch10"
    "SimNPO open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5"
    "RMU open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer10_scoeff100_epoch5"
    "UNDIAL open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha2_epoch10"
)


## TOFU
model="Llama-3.2-1B-Instruct"
# model="Llama-3.1-8B-Instruct"
splits=(
    # "forget01 holdout01 retain99"
    # "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for tr in "${trainers[@]}"; do
        model_path=open-unlearning/tofu_${model}_full
        trainer=$(echo $tr | cut -d' ' -f1)
        ckpt=$(echo $tr | cut -d' ' -f2)

        task_name=tofu_${model}_${forget_split}_${trainer}
        # _standard_42
        # task_name=none/tofu_${model}_${forget_split}_${trainer}_standard_42

        if [ ! -f saves/sample_difficulty/tofu/${model}/${forget_split}/${trainer}/loss.pt ]; then
            echo "TOFU ${forget_split} Difficulty Not Found"

            if [ "$trainer" = "Original" ]; then
                CUDA_VISIBLE_DEVICES=2 python src/compute_sample_difficulty.py \
                experiment=eval/tofu/default.yaml \
                task_name=compute_sample_difficulty/tofu/${task_name} \
                forget_split=${forget_split} \
                holdout_split=${holdout_split} \
                model=${model} \
                model.model_args.pretrained_model_name_or_path=${model_path}

            else
                CUDA_VISIBLE_DEVICES=2 python src/compute_sample_difficulty.py \
                experiment=eval/tofu/default.yaml \
                task_name=compute_sample_difficulty/tofu/${task_name} \
                forget_split=${forget_split} \
                holdout_split=${holdout_split} \
                model=${model} \
                model.model_args.pretrained_model_name_or_path=${ckpt}
                # model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name}
            fi
        fi
    done
done


## MUSE
# model=Llama-2-7b-hf
# data_splits=(
#     "News"
#     "Books"
# )

# models=(
#     Llama-2-7b-hf
# )
# data_splits=(
#     "News"
#     "Books"
# )

# for data_split in "${data_splits[@]}"; do
#     for trainer in "${trainers[@]}"; do
#         task_name=muse_${model}_${data_split}_${trainer}_standard_42

#         if [ ! -f saves/sample_difficulty/muse/${data_split}/${trainer}/loss.pt ]; then
#             echo "MUSE ${data_split} Difficulty Not Found"

#             if [ "$trainer" = "Original" ]; then
#                 CUDA_VISIBLE_DEVICES=2 python src/compute_sample_difficulty.py \
#                 experiment=eval/muse/default.yaml \
#                 task_name=compute_sample_difficulty/muse/${data_split} \
#                 data_split=${data_split} \
#                 model=${model}
            
#             else
#                 CUDA_VISIBLE_DEVICES=2 python src/compute_sample_difficulty.py \
#                 experiment=eval/muse/default.yaml \
#                 task_name=compute_sample_difficulty/muse/${data_split} \
#                 data_split=${data_split} \
#                 model=${model} \
#                 model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name}
#             fi
#         fi
#     done
# done


# ## WMDP
# model=zephyr-7b-beta
# data_splits=(
#     # "cyber"
#     "bio"
# )

# for data_split in "${data_splits[@]}"; do
#     for trainer in "${trainers[@]}"; do
#         task_name=wmdp_${model}_${data_split}_${trainer}_standard_42

#         if [ ! -f saves/sample_difficulty/wmdp/${data_split}/${trainer}/loss.pt ]; then
#             echo "WMDP ${data_split} Difficulty Not Found"

#             if [ "$trainer" = "Original" ]; then
#                 CUDA_VISIBLE_DEVICES=2 python src/compute_sample_difficulty.py \
#                 experiment=eval/wmdp/default.yaml \
#                 task_name=compute_sample_difficulty/wmdp/${data_split} \
#                 data_split=${data_split} \
#                 model=${model}

#             else
#                 CUDA_VISIBLE_DEVICES=2 python src/compute_sample_difficulty.py \
#                 experiment=eval/wmdp/default.yaml \
#                 task_name=compute_sample_difficulty/wmdp/${data_split} \
#                 data_split=${data_split} \
#                 model=${model} \
#                 model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name}
#             fi
#         fi
#     done
# done
