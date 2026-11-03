import pytest

from llm_platform.benchmark import BenchmarkConfig
from llm_platform.contracts import GenerationConfig


@pytest.mark.parametrize("settings", [{"max_tokens": 0}, {"max_tokens": True}, {"max_tokens": 1.5},
                                      {"max_tokens": 32769}, {"seed": -1}, {"seed": True},
                                      {"seed": 2**32}, {"temperature": float("nan")},
                                      {"temperature": True}, {"stop": "STOP"}, {"stop": ("",)},
                                      {"stop": tuple("abcdef")}, {"stop": ("x" * 257,)}])
def test_invalid_direct_generation_settings_fail_before_inference(settings):
    with pytest.raises(ValueError):
        GenerationConfig(**settings)


def test_valid_generation_settings_freeze_mutable_stop_input():
    stops = ["END"]
    config = GenerationConfig(stop=stops)
    stops.append("OTHER")
    assert config.stop == ("END",)


@pytest.mark.parametrize("settings", [{"requests": True}, {"requests": 1.5}, {"concurrency": 1.5},
                                      {"warmup": 0.5}, {"repetition": -1}, {"seed": -1},
                                      {"seed": True}, {"temperature": True}, {"request_rate": True},
                                      {"timeout_seconds": True}, {"compute_price_per_hour": True}])
def test_invalid_benchmark_settings_fail_before_creating_output(settings):
    with pytest.raises(ValueError):
        BenchmarkConfig(**settings)
