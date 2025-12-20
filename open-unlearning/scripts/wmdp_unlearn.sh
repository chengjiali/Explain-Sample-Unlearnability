#!/bin/bash


export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

trainers=(
    # "Original"
    "GradDiff"
    "NPO"
    "SimNPO"
    "RMU"
    "UNDIAL"
)
models=(
    # zephyr-7b-beta
    # Phi-3.5-mini-instruct
    phi-1_5
)
data_splits=(
    # "cyber"
    "bio"
)


per_device_train_batch_size=4 # on two gpus would make effective batch size 32
gradient_accumulation_steps=8


for data_split in "${data_splits[@]}"; do
    for model in "${models[@]}"; do
        for trainer in "${trainers[@]}"; do
            task_name=wmdp_${model}_${data_split}_${trainer}

            if [ "$trainer" = "Original" ]; then
                if [ "$model" = "phi-1_5" ]; then
                    ckpt='microsoft/Phi-3.5-mini-instruct'
                else
                    ckpt='microsoft/phi-1_5'
                fi
            else
                ckpt=saves/unlearn/${task_name}
            fi

            TRAIN_CMD="CUDA_VISIBLE_DEVICES=2 python \
            src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/wmdp/default.yaml \
            trainer=${trainer} \
            model=${model} \
            data_split=${data_split} \
            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            trainer.args.eval_strategy=no"

            EVAL_CMD="CUDA_VISIBLE_DEVICES=2 python src/eval.py \
            experiment=eval/wmdp/default.yaml \
            model=${model} \
            data_split=${data_split}"

            DIF_CMD="CUDA_VISIBLE_DEVICES=2 python src/compute_sample_difficulty.py \
            experiment=eval/wmdp/default.yaml \
            data_split=${data_split} \
            model=${model}"


            if [ "$trainer" != "Original" ]; then
                if [ ! -f saves/unlearn/"${task_name}"/model.safetensors ] && [ ! -f saves/unlearn/"${task_name}"/model.safetensors.index.json ]; then
                    echo "${task_name}" "Model Not Found"

                    eval ${TRAIN_CMD} \
                    task_name=${task_name} \
                    trainer.cl.method="none"
                fi
            fi

            if [ ! -f saves/unlearn/"${task_name}"/evals/LMEval_SUMMARY.json ]; then
                echo "${task_name}" "Eval Not Found"

                eval ${EVAL_CMD} \
                task_name=${task_name} \
                model.model_args.pretrained_model_name_or_path=${ckpt} \
                paths.output_dir=saves/unlearn/${task_name}/evals
            fi

            if [ ! -f saves/sample_difficulty/wmdp/${data_split}/${trainer}/loss.pt ]; then
                echo "${task_name}" "Loss Not Found"

                eval ${DIF_CMD} \
                task_name=${task_name} \
                model.model_args.pretrained_model_name_or_path=${ckpt}
            fi
        done
    done
done
