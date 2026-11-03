# Astra medium adversarial review — adapter precision

Verdict at discovery: **P1 fairness blocker**.

Installed PEFT source and the reviewer's CPU-only miniature Qwen probe confirmed
that default `PeftModel.from_pretrained(..., is_trainable=False)` upcasts LoRA
parameters to FP32. Its forward path casts adapter inputs to that dtype. The
reference backend did not use an inference autocast context. Installed vLLM
resolves LoRA `auto` to the BF16 base dtype and casts loaded adapter tensors.

Therefore identical serialized FP32 adapter files and a BF16 base manifest did
not establish equal adapter compute precision. The original adapter performance
measurements must remain exploratory and cannot support a fair engine speedup.

Required correction: disable PEFT adapter upcasting during inference, verify and
record actual adapter parameter precision, explicitly configure vLLM LoRA BF16,
reject missing or mismatched precision in comparisons, preserve all original
observations, and rerun affected adapter conditions. Base-only measurements,
training, and the historical internally controlled quality experiment remain valid.

Final resolution and independent re-review are required after aligned real GPU
measurements. This record does not claim those measurements or review have passed.
