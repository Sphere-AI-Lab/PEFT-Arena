<p align="center">
  <img src="assets/PEFT-Arena.png" width="80%" alt="PEFT-Arena"/>
</p>

# PEFT-Arena

Official code release for the PEFT-Arena paper:

**PEFT-Arena: Understanding Parameter-Efficient Finetuning from a Stability-Plasticity Perspective**

Yangyi Huang, Ruotian Peng, Zeju Qiu, Jiale Kang, Yandong Wen, Bernhard Schölkopf, Weiyang Liu

Project page: https://spherelab.ai/PEFT-Arena  
GitHub: https://github.com/Sphere-AI-Lab/PEFT-Arena  
Project site source: [docs/](docs/)

## Abstract

PEFT-Arena studies parameter-efficient finetuning through the stability-plasticity dilemma: how much a post-trained model improves on the target domain, and how much of its pretrained general capability it retains.

This repository contains the official training, evaluation, and analysis code for the paper's main experimental workflows:

- supervised finetuning (SFT)
- reinforcement learning with verifiable rewards (RLVR / GRPO)
- evaluation on target-domain and general-retention benchmarks
- spectral retention-adaptation profiling and plotting

The benchmark covers two target domains:

- mathematical reasoning
- medical reasoning

and measures general capability retention on:

- BBH
- IFEval
- NQ

The paper reports experiments on `Qwen2.5-7B` and `Llama3.2-3B-Instruct`, comparing full finetuning and representative PEFT families including LoRA variants, OFT, IA3, VeRA, MiSS, and KeepLoRA.

## What this repo provides

- post-training with SFT and RLVR
- evaluation on target-domain and general-retention benchmarks
- spectral analysis and figure generation used in the paper
- Checkpoints and data

## Highlights

- A unified benchmark for evaluating PEFT beyond downstream accuracy alone.
- A stability-plasticity view of SFT and RLVR post-training.
- Spectral analysis tools for studying retention and adaptation structure in weight updates.
- Reproducible training and evaluation entrypoints through a single CLI in `run.py`.

## What Is Included

- `run.py`: unified CLI for training, evaluation, and adapter merge
- `train/`: SFT and RL training wrappers plus PEFT-Arena-owned trainer code
- `eval/`: math, medical, and general evaluation pipelines
- `tools/`: checkpoint preparation, merge, spectral analysis, and plotting
- `third_party/math_eval` and `third_party/med_eval`: bundled target-domain evaluation code
- `third_party/opencompass` and `third_party/verl`: external dependencies used by general evaluation and RL training
- `docs/`: project website / GitHub Pages source

## Installation

Run commands from the repository root.

You should start from a Python environment with a compatible CUDA / PyTorch stack. The release setup script installs the PEFT-Arena-side dependencies on top of that environment.

Typical setup:

```bash
bash setup_env.sh
```

This script:

- validates and backfills required `third_party/` components
- installs training and evaluation dependencies
- installs the patched `math_eval/latex2sympy` package with `antlr4-python3-runtime==4.9.3`
- installs `OpenCompass`, `VeRL`, `vllm`, `human-eval`, and `evalplus`

If you only want to fetch or validate the third-party trees:

```bash
bash setup_third_party.sh
```

Notes:

- `third_party/math_eval` and `third_party/med_eval` are included in this release.
- `third_party/human-eval` is not tracked in git; `setup_third_party.sh` will copy or fetch it when needed.
- OpenCompass benchmark data is not bundled. Some datasets download automatically on first use, while others such as `IFEval` and `mmlu` still require local dataset preparation under `third_party/opencompass/data`.

## Benchmark Protocol

### Target-domain evaluation

- **Math**: `math500`, `amc23`, `aime24`
- **Medical**: packaged `med_eval` benchmark set including medical QA and reasoning tasks used in the paper

### General retention evaluation

- `bbh`
- `ifeval_nq`
- extended wrappers for `humaneval`, `hellaswag`, `winogrande`, `mmlu`, `arc`, `gsm8k`, and `xcopa`

### Training settings

- **SFT**
  - math data: filtered 50k samples from OpenR1-Math
  - medical data: 23k samples from MedThink
- **RLVR**
  - PEFT-Arena RL training with GRPO
  - release code keeps the RL dataset and async rollout / agent-loop path aligned with the current `peft_arena` implementation

## Quick Start

### 1. SFT

```bash
python run.py train sft \
  --model Qwen/Qwen2.5-7B \
  --adapter lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --data_train data/openr1-50k/train.parquet \
  --data_val data/openr1-50k/test.parquet \
  --output_dir checkpoints/sft/math/qwen2.5-7b/lora-r16
```

### 2. RLVR / GRPO

```bash
python run.py train rl \
  --model Qwen/Qwen2.5-7B \
  --adapter oft \
  --oft_block_size 32 \
  --data_train data/openr1-50k/train.parquet \
  --data_val data/openr1-50k/test.parquet \
  --output_dir checkpoints/rl/math/qwen2.5-7b/oft-b32
```

### 3. Evaluation

Evaluate one checkpoint on all supported domains:

```bash
python run.py eval \
  --checkpoint_path checkpoints/sft/math/qwen2.5-7b/lora-r16/global_step_780 \
  --domain all
```

Evaluate general-retention benchmarks only:

```bash
python run.py eval \
  --checkpoint_path checkpoints/sft/med/qwen2.5-7b/oft-b16/global_step_364 \
  --domain general \
  --benchmarks bbh,ifeval_nq,humaneval
```

Convenience wrappers remain available:

```bash
bash eval/eval_math.sh --checkpoint_path <ckpt>
bash eval/eval_med.sh --checkpoint_path <ckpt>
bash eval/eval_general.sh --checkpoint_path <ckpt>
```

### 4. Checkpoint export and merge

Prepare a training checkpoint for evaluation:

```bash
python tools/prepare_eval_checkpoint.py \
  --checkpoint_path checkpoints/sft/med/qwen2.5-7b/oft-b16/global_step_364
```

Merge a PEFT adapter into a standalone Hugging Face checkpoint:

```bash
python run.py merge \
  --adapter_path checkpoints/sft/math/qwen2.5-7b/lora-r16/global_step_780 \
  --output_path checkpoints/sft/math/qwen2.5-7b/lora-r16/global_step_780_merged
```

### 5. Result summarization

```bash
python eval/summarize_results.py --results_dir results
python eval/extract_new_benchmark_summary.py \
  --input results/summary.csv \
  --output results/new_benchmark_summary.csv
```

### 6. Spectral analysis

Run spectral analysis for one base / finetuned pair:

```bash
python tools/spectral_analysis.py \
  --base_model Qwen/Qwen2.5-7B \
  --finetuned_model checkpoints/sft/math/qwen2.5-7b/lora-r8/global_step_780 \
  --output_dir analysis/math/sft/qwen2.5-7b/lora-r8/global_step_780 \
  --layers 18 \
  --modules down_proj
```

Plot the analysis outputs:

```bash
python tools/plot_spectral_analysis.py \
  --input_dirs analysis/math/sft/qwen2.5-7b/full/global_step_780 analysis/math/sft/qwen2.5-7b/oft-b32/global_step_780 analysis/math/sft/qwen2.5-7b/lora-r8/global_step_780 \
  --labels SFT-FullFT SFT-OFT-b32 SFT-LoRA-r8 \
  --output_dir analysis/plot_sft_spectrum \
  --plot_type curves \
  --layer_names "model.layers.18.mlp.down_proj.weight" \
  --log_scale
```

Composite and utility plots:

```bash
python tools/plot_spectral_composite.py \
  --output analysis/plot_composite/composite_layer18_mlp_down_proj.pdf

python tools/plot_spectral_method_grid.py \
  --output analysis/plot_composite/method_grid_layer18_mlp_down_proj.pdf

python tools/compare_model_norms.py \
  --base-model Qwen/Qwen2.5-7B \
  --sft-model checkpoints/sft/math/qwen2.5-7b/lora-r8/global_step_780 \
  --output analysis/model_norms/qwen2.5-7b_lora-r8
```

## Resources

[Experiment Checkpoints](https://huggingface.co/YangyiH/PEFT-Arena-SFT/)

[Training Data](https://huggingface.co/datasets/SphereLab/PEFTArena-data)

## Repository Layout

```text
peft_arena_release/
├── run.py
├── setup_env.sh
├── setup_third_party.sh
├── configs/
├── docs/
├── train/
├── eval/
├── tools/
├── tests/
└── third_party/
```

## Citation

If you find PEFT-Arena useful in your research, please cite:

```bibtex
@article{peftarena2026,
      title={PEFT-Arena: Understanding Parameter-Efficient Finetuning from a Stability-Plasticity Perspective},
      author={Yangyi Huang and Ruotian Peng and Zeju Qiu and Jiale Kang and Yandong Wen and Bernhard Schölkopf and Weiyang Liu},
      journal={arXiv preprint arXiv:2605.28819},
      year={2026},
}
```

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
