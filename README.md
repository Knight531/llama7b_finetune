# Alpaca-LoRA Finetune on NL2SQL

This repository contains code for reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) results using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).

This repository is forked from https://huggingface.co/decapoda-research/llama-7b-hf/tree/main.

It requires at least 10 gigabytes VRAM memory. We run this program on A100-80G.

### Local Setup

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

1. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

### Training (`finetune.py`)

This file contains a straightforward application of PEFT to the LLaMA model, also include the dataset we use to finetune the model.

File "apacha_train.json" corresponding to 1000 rows dataset.

File "apacha_train_7000.json" corresponding to 7000 rows dataset.

We suggest you to tweak our hyperparameters:

```bash
nohup python \
finetune.py \
--base_model 'decapoda-research/llama-7b-hf' \
--data_path '/home/gpt/data/apacha_train.json' \
--output_dir './my_model_save' \
--batch_size 128 \
--micro_batch_size 2 \
--num_epochs 2 \
--learning_rate 3e-4 \
1>train.log 2>&1 &
```

When you first run this instruct, it will take you a long time to download llama-7b-hf model.

### Inference (`generate.py`)

Run the following code to infer on the web.

```bash
python generate.py \
--base_model "decapoda-research/llama-7b-hf" \
--lora_weights './my_model_save' \
--load_8bit
```

