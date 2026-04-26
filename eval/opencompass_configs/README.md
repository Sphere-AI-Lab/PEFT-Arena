PEFTArena stores local copies of a small set of OpenCompass dataset configs here
so evaluation behavior does not depend on patching `third_party/opencompass` or an
installed OpenCompass checkout.

Current local dataset sources:
- `datasets/IFEval/IFEval_gen_3321a3.py`: copied from OpenCompass.
- `datasets/nq/nq_gen_c788f6.py`: copied from `opencompass_ours` to preserve the
  NQ prompt variant that asks the model to answer with `"The answer is"`.
- `datasets/bbh/bbh_gen_5b92b0.py`: copied from `opencompass_ours` to preserve
  the BBH dataset inferencer `max_out_len=512` behavior.
- `datasets/bbh/lib_prompt/`: copied with the BBH config because the config reads
  these prompt files via `__file__`-relative paths.

Additional OpenCompass datasets such as `humaneval`, `hellaswag`,
`winogrande`, `mmlu`, `ARC_c`, `gsm8k`, and `XCOPA` are inlined directly from
the bundled OpenCompass configs at generation time and do not require local
copies here.
