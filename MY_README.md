0. Install Requirements

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

1. Training Data Generation

1.1 Generating Text-Only Data for Initial Training

```bash
python -m vispec.ge_data.allocation_llava_shargpt \
  --outdir=/home/youngmin/workspace/ViSpec/datasets/llava_shargpt \
  --start=0 \
  --end=67999 \
  --model=/data/youngmin/models/llava-v1.6-vicuna-7b-hf
```
기존 --end=67999

1.2 Generating Multimodal Data for ViSpec Training

```bash
python -m vispec.ge_data.allocation_llava_pretrain_gen \
  --outdir=/home/youngmin/workspace/ViSpec/datasets/llava_pretrain_gen \
  --start=0 \
  --end=67999 \
  --model=/data/youngmin/models/llava-v1.6-vicuna-7b-hf
```
기존 --end=67999

2. Model Training

2.1 Initial Training
```bash
accelerate launch --multi_gpu \
  -m --mixed_precision=bf16 \
  vispec.train.main \
  --cpdir=/data/youngmin/checkpoints/pre \
  --basepath=/data/youngmin/models/llava-v1.6-vicuna-7b-hf \
  --begin-epoch=0 \
  --bs=1 \
  --configpath=/home/youngmin/workspace/ViSpec/vispec/train/llava_1.6_7B_config.json \
  --lr=3e-6 \
  --max-len=4096 \
  --num-workers=8 \
  --tmpdir=/home/youngmin/workspace/ViSpec/datasets/llava_shargpt
```

2.2 Training with ViSpec
```bash
accelerate launch --multi_gpu \
  -m --mixed_precision=bf16 \
  vispec.train.main_mtp \
  --cpdir=/data/youngmin/checkpoints/fine \
  --basepath=/data/youngmin/models/llava-v1.6-vicuna-7b-hf \
  --begin-epoch=0 \
  --bs=1 \
  --configpath=/home/youngmin/workspace/ViSpec/vispec/train/llava_1.6_7B_config.json \
  --loadpath=/data/youngmin/checkpoints/pre/state_20/model.safetensors \
  --lr=3e-6 \
  --max-len=4096 \
  --mtp-steps=1 \
  --num-q=2 \
  --num-workers=8 \
  --tmpdir=/home/youngmin/workspace/ViSpec/datasets/llava_pretrain_gen \
  --use-ours=True
```

3. Evaluation

3.1 Baseline Speed Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m vispec.evaluation.gen_baseline_answer_mmvet \
  --base-model-path=/data/youngmin/models/llava-v1.6-vicuna-7b-hf \
  --bench-name=/data/youngmin/results/test \
  --spec-model-path=/data/youngmin/models/ViSpec-llava-v1.6-vicuna-7b-hf \
  --temperature=0.0
```
--model-id test \

3.2 Speculative Decoding Speed Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m vispec.evaluation.gen_spec_answer_mmvet \
  --base-model-path=/data/youngmin/models/llava-v1.6-vicuna-7b-hf \
  --bench-name=/data/youngmin/results/test \
  --spec-model-path=/data/youngmin/checkpoints/fine/state_20 \
  --num-q=2 \
  --depth=3 \
  --top-k=8 \
  --total-token=30 \
  --use-ours=True \
  --temperature=0.0
```
--model-id test \
