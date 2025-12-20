#!/bin/bash


export WANDB_MODE=disabled
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

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
models=(
    "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
)
splits=(
    # "forget01 holdout01 retain99"
    # "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)


per_device_train_batch_size=4 # on two gpus would make effective batch size 32
gradient_accumulation_steps=8


for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for tr in "${trainers[@]}"; do
            trainer=$(echo $tr | cut -d' ' -f1)
            ckpt=$(echo $tr | cut -d' ' -f2)

            task_name=tofu_${model}_${forget_split}_${trainer}
            model_path=open-unlearning/tofu_${model}_full

            if [ "$trainer" = "Original" ]; then
                ckpt=${model_path}
            fi

            TRAIN_CMD="CUDA_VISIBLE_DEVICES=2 python \
            src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default.yaml \
            trainer=${trainer} \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            trainer.args.eval_strategy=no"

            EVAL_CMD="CUDA_VISIBLE_DEVICES=2 python src/eval.py \
            experiment=eval/tofu/default.yaml \
            model=${model} \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

            DIF_CMD="CUDA_VISIBLE_DEVICES=2 python src/compute_sample_difficulty.py \
            experiment=eval/tofu/default.yaml \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            model=${model}"

            # if [ "$trainer" != "Original" ]; then
            #     if [ ! -f saves/unlearn/"${task_name}"/model.safetensors ] && [ ! -f saves/unlearn/"${task_name}"/model.safetensors.index.json ]; then
            #         echo "${task_name}" "Model Not Found"

            #         eval ${TRAIN_CMD} \
            #         task_name=${task_name} \
            #         trainer.cl.method="none"
            #     fi
            # fi

            if [ ! -f saves/unlearn/"${task_name}"/evals/TOFU_SUMMARY.json ]; then
                echo "${task_name}" "Eval Not Found"

                eval ${EVAL_CMD} \
                task_name=${task_name} \
                model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
                paths.output_dir=saves/unlearn/${task_name}/evals
            fi

            if [ ! -f saves/sample_difficulty/tofu/${model}/${forget_split}/${trainer}/loss.pt ]; then
                echo "${task_name}" "Loss Not Found"

                eval ${DIF_CMD} \
                task_name=${task_name} \
                model.model_args.pretrained_model_name_or_path=${ckpt}
            fi
        done
    done
done
