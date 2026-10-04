

# Circuit-Guided Unlearning Difficulty



## How to Use Existing CUD

Please download the CUD scores. Each file contains the score for each sample. 
- [TOFU](./open-unlearning/saves/cud_tofu/cud_cosine.pt)
- MovieLens (coming soon)


## How to Train CUD

#### 1. Install

```bash
pip install -r requirements.txt
```

#### 2. Select Easy/Hard Samples

```bash
python scripts/select_easy_hard.py \
  --score-path saves/cud/tofu/all.txt \
  --num-samples 50 \
  --easy low \
  --output-dir outputs/tofu_cud_selection
```

For post-unlearning loss files like the old `find_easy_hard_samples.py` flow, higher loss usually means the sample was easier to forget:

```bash
python scripts/select_easy_hard.py \
  --score-path ../open-unlearning/saves/sample_difficulty/tofu/Llama-3.2-1B-Instruct/forget10/GradDiff/loss.pt \
  --score-path ../open-unlearning/saves/sample_difficulty/tofu/Llama-3.2-1B-Instruct/forget10/NPO/loss.pt \
  --score-path ../open-unlearning/saves/sample_difficulty/tofu/Llama-3.2-1B-Instruct/forget10/SimNPO/loss.pt \
  --score-path ../open-unlearning/saves/sample_difficulty/tofu/Llama-3.2-1B-Instruct/forget10/RMU/loss.pt \
  --score-path ../open-unlearning/saves/sample_difficulty/tofu/Llama-3.2-1B-Instruct/forget10/UNDIAL/loss.pt \
  --num-samples 50 \
  --min-appearances 3 \
  --easy high \
  --output-dir outputs/tofu_loss_selection
```

Outputs:

- `selected_samples.json`
- `easy_ids.txt`
- `hard_ids.txt`
- `ranked_scores.csv`

#### 3. Split TOFU

Split `locuslab/TOFU` `forget10` into easy/hard forget sets. Optionally add `retain90` to each retain set so each unlearning run retains the original retain pool plus forget-pool examples not selected for removal.

```bash
python scripts/split_tofu.py \
  --selection outputs/tofu_cud_selection/selected_samples.json \
  --dataset locuslab/TOFU \
  --name forget10 \
  --split train \
  --retain-dataset locuslab/TOFU \
  --retain-name retain90 \
  --retain-split train \
  --output-dir data/tofu_easy_hard
```

Outputs:

- `forget_easy.jsonl`
- `forget_hard.jsonl`
- `retain_easy.jsonl`
- `retain_hard.jsonl`
- `manifest.json`

#### 4. Run Unlearning

The runner launches the existing `open-unlearning` Hydra entrypoint with local JSONL split files:

```bash
python scripts/run_unlearning.py \
  --open-unlearning-dir open-unlearning \
  --splits-dir data/tofu_easy_hard \
  --model Llama-3.2-1B-Instruct \
  --model-path ../open-unlearning/open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  --trainer GradDiff \
  --device 0
```
