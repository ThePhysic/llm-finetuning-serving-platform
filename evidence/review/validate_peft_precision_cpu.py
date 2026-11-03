"""Installed-library regression proof on a tiny CPU model; never GPU evidence."""
import importlib.metadata
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

# PEFT may choose CUDA while loading safetensors even for a CPU base. Hide all
# devices before importing Torch so this protocol proof cannot create a context.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from llm_platform.backends.precision import adapter_storage_dtype, inspect_lora_compute_dtype


def base():
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=2)
    return Qwen2ForCausalLM(config).to(device="cpu", dtype=torch.bfloat16)


with TemporaryDirectory(prefix="llm-lora-precision-cpu-") as temporary:
    path = Path(temporary)
    trained = get_peft_model(base(), LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"],
                                               task_type="CAUSAL_LM", bias="none"))
    trained.save_pretrained(path, safe_serialization=True)
    saved_dtype = adapter_storage_dtype(path / "adapter_model.safetensors")
    default = PeftModel.from_pretrained(base(), path, is_trainable=False)
    default_dtypes = sorted({str(p.dtype) for name, p in default.named_parameters() if "lora_" in name})
    corrected = PeftModel.from_pretrained(base(), path, is_trainable=False, autocast_adapter_dtype=False)
    corrected_dtype = inspect_lora_compute_dtype(corrected, "bfloat16")
    assert saved_dtype == "float32"
    assert default_dtypes == ["torch.float32"]
    assert corrected_dtype == "bfloat16"
    assert all(p.device.type == "cpu" for p in corrected.parameters())
    assert not torch.cuda.is_initialized()
    print(json.dumps({"peft_version": importlib.metadata.version("peft"),
        "torch_version": torch.__version__, "all_model_tensors_on_cpu": True,
        "CUDA_context_initialized": torch.cuda.is_initialized(),
        "adapter_storage_dtype": saved_dtype, "default_PEFT_adapter_runtime_dtypes": default_dtypes,
        "autocast_adapter_dtype_false_runtime_dtype": corrected_dtype}, indent=2))
