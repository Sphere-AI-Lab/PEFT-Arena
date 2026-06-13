# Single-Checkpoint Geometry Diagnostics

This release includes lightweight scripts for running the internal geometry
diagnostics used in PEFT-Arena on one checkpoint at a time.  These scripts are
intended for reproducibility and debugging, not for the paper's full
all-checkpoint batch pipeline.

The scripts do not read PEFT-Arena's historical result tables, checkpoint
manifests, or local absolute paths.  You provide:

- a pretrained base model;
- one finetuned full-model checkpoint or PEFT adapter checkpoint;
- prompt files for the distributions you want to probe;
- the module(s) to analyze.

## Prompt Files

Prompt files can be `.txt`, `.jsonl`, or `.json`.

For `.txt`, each non-empty line is treated as one prompt.  For `.jsonl` or
`.json`, pass `--text_key` if the text field is not one of:

```text
prompt, input, question, query, problem, instruction, text
```

Use evaluation-aligned prompts whenever possible.  The scripts only run forward
passes and do not generate answers.

## Capability-Conditioned Drift

`tools/csd_single_checkpoint.py` computes capability-conditioned drift (CSD)
for one linear module.  It collects input activations from the base model and
measures the effect of the effective weight update:

```text
CSD_abs(D) = E ||Delta W h||_2^2
CSD_rel(D) = E ||Delta W h||_2^2 / (E ||W0 h||_2^2 + eps)
CSD_update_norm(D) =
    E ||Delta W h||_2^2 / ((||Delta W||_F^2 + eps) * (E ||h||_2^2 + eps))
```

Example:

```bash
python tools/csd_single_checkpoint.py \
  --base_model /path/to/Qwen2.5-7B \
  --checkpoint /path/to/adapter_or_merged_checkpoint \
  --general_prompts data/diagnostics/general_prompts.jsonl \
  --target_prompts data/diagnostics/math_prompts.jsonl \
  --module model.layers.18.mlp.down_proj \
  --output_dir results/diagnostics/qwen_sft_math_lora_r8_csd \
  --device cuda \
  --batch_size 1 \
  --max_examples 300 \
  --max_tokens 4096
```

Outputs:

```text
csd_summary.csv
csd_summary.json
csd_dataset_metrics.csv
```

## Activation-Space Geometry

`tools/activation_geometry_single_checkpoint.py` compares base-model and
finetuned-model full-forward module outputs on the same prompts.  It reports:

- pointwise cosine and angular drift;
- norm drift;
- pairwise cosine Gram distortion;
- raw residual;
- orthogonal Procrustes residual;
- Procrustes improvement;
- linear CKA;
- PCA subspace distances.

Example for one module:

```bash
python tools/activation_geometry_single_checkpoint.py \
  --base_model /path/to/Qwen2.5-7B \
  --checkpoint /path/to/adapter_or_merged_checkpoint \
  --general_prompts data/diagnostics/general_prompts.jsonl \
  --target_prompts data/diagnostics/math_prompts.jsonl \
  --modules model.layers.18.mlp.down_proj \
  --output_dir results/diagnostics/qwen_sft_math_lora_r8_activation_geometry \
  --device cuda \
  --batch_size 1 \
  --max_examples 300 \
  --max_tokens 4096
```

Example for the eight module locations used in the paper's pilot:

```bash
python tools/activation_geometry_single_checkpoint.py \
  --base_model /path/to/Qwen2.5-7B \
  --checkpoint /path/to/adapter_or_merged_checkpoint \
  --general_prompts data/diagnostics/general_prompts.jsonl \
  --target_prompts data/diagnostics/math_prompts.jsonl \
  --modules paper8 \
  --output_dir results/diagnostics/qwen_sft_math_lora_r8_activation_geometry_paper8
```

Output:

```text
activation_geometry_summary.csv
activation_geometry_summary.json
```

## Notes

- If `--checkpoint` points to a PEFT adapter directory, the scripts load a fresh
  base model and merge the adapter in memory.
- For full finetuning checkpoints, pass the full model directory directly.
- CSD uses base-model activations for comparability.
- Activation geometry uses full-forward base and finetuned activations.
- Large models can be memory intensive.  Reduce `--max_tokens`, `--max_length`,
  or the number of modules if needed.
