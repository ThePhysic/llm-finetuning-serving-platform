"""Genuine model execution backends; no generated-output fixtures in production."""
from llm_platform.backends.transformers_backend import TransformersBackend
from llm_platform.backends.vllm_backend import VLLMBackend

__all__ = ["TransformersBackend", "VLLMBackend"]
