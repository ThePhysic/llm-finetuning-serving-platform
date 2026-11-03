"""Command-line entry points; heavy GPU libraries are loaded only when required."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help=__doc__)
data_app = typer.Typer(no_args_is_help=True)
registry_app = typer.Typer(no_args_is_help=True)
review_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(registry_app, name="registry")
app.add_typer(review_app, name="review")


def emit(value):
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str))


@data_app.command("generate")
def generate(output: Path):
    """Write the reproducible, authored synthetic corpus; no external API is used."""
    from .synthetic import write_synthetic
    if output.exists():
        raise typer.BadParameter("Use a new output path to preserve the existing corpus")
    output.parent.mkdir(parents=True, exist_ok=True)
    emit({"records": write_synthetic(output), "output": output})


@data_app.command("prepare")
def prepare(source: Path, output: Path, seed: int = 42):
    from .data import prepare_dataset
    emit(prepare_dataset(source, output, seed=seed))


@data_app.command("verify")
def verify_data(dataset: Path):
    from .data import verify_dataset
    emit(verify_dataset(dataset))


@app.command("train")
def train_command(config: Path, dataset: Path, output: Path):
    """Run real PyTorch/PEFT SFT, using only verified train/validation files."""
    from .data import verify_dataset
    from .training import TrainConfig, train
    manifest = verify_dataset(dataset)
    settings = TrainConfig.from_json(config)
    if settings.dataset_id != manifest["dataset_id"]:
        raise typer.BadParameter("Training config dataset_id does not match the frozen manifest")
    result = train(settings, dataset / "train.jsonl", dataset / "validation.jsonl", output)
    emit({key: value for key, value in result.items() if key != "metrics"})


@app.command("package-base")
def package_base(output: Path, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 revision: str = "main", dtype: str = "bfloat16", backend: str = "transformers"):
    """Resolve an immutable base revision and package its tokenizer and provenance."""
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from .registry import create_package
    from .training import _code_identity
    model = HfApi().model_info(model_id, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model.sha, trust_remote_code=False)
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template.strip():
        raise ValueError("A nonempty string chat template is required; named templates are unsupported")
    code_revision, code_dirty = _code_identity()
    with tempfile.TemporaryDirectory(prefix="llm-base-") as temporary:
        source = Path(temporary)
        tokenizer.save_pretrained(source / "tokenizer")
        metadata = {"schema_version": 1, "kind": "base",
            "base_model": {"id": model_id, "revision": model.sha},
            "tokenizer": {"id": model_id, "revision": model.sha, "path": "tokenizer",
                          "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()},
            "adapter": None, "runtime": {"backend": backend, "quantization": "none", "dtype": dtype},
            "provenance": {"dataset_id": None, "experiment_id": "base-package", "code_revision": code_revision,
                           "seed": 42, "code_dirty": code_dirty,
                           "license": (model.card_data or {}).get("license")},
            "evaluation": {"report": None, "limitations": ["Base package; evaluate before a deployment decision."]}}
        emit(create_package(source, output, metadata))


@registry_app.command("register")
def register(package: Path, registry: Path = Path("registry")):
    from .registry import Registry
    emit({"artifact_id": Registry(registry).register(package)})


@registry_app.command("list")
def list_models(registry: Path = Path("registry")):
    from .registry import Registry
    emit(Registry(registry).list())


@registry_app.command("verify")
def verify_model(artifact_id: str, registry: Path = Path("registry")):
    from .registry import Registry
    package, manifest = Registry(registry).verify(artifact_id)
    emit({"package": package, "manifest": manifest})


@app.command("serve")
def serve(registry: Path = Path("registry"), model: str | None = None,
          host: str = "127.0.0.1", port: int = 8000, concurrency: int = 1,
          max_queue: int = 8, max_input_tokens: int = 2048, max_output_tokens: int = 512,
          timeout: float = 120.0):
    """Start the gateway. Non-local binding requires distinct access/admin env keys."""
    import uvicorn
    from .serving import Settings, create_app
    settings = Settings(registry_path=registry, initial_model=model, host=host,
        api_key=os.environ.get("LLM_API_KEY"), admin_key=os.environ.get("LLM_ADMIN_KEY"),
        max_concurrency=concurrency, max_queue=max_queue, max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens, request_timeout_seconds=timeout)
    uvicorn.run(create_app(settings), host=host, port=port, access_log=False)


@app.command("evaluate")
def evaluate(package: Path, dataset: Path, output: Path, variant: str = "zero_shot",
             split: str = "test", max_tokens: int = 320, include_regressions: bool = True):
    """Persist actual generations and scores; validation and test results stay separate."""
    from .contracts import GenerationConfig
    from .data import load_records, verify_dataset
    from .evaluation import run_evaluation
    from .registry import verify_package
    from .serving import default_backend_factory
    if split not in {"test", "validation"}:
        raise typer.BadParameter("Evaluation split must be test or validation")
    if variant not in {"zero_shot", "prompted", "lora"}:
        raise typer.BadParameter("Unknown prompt variant")
    data = verify_dataset(dataset)
    model = verify_package(package)
    if (variant == "lora") != (model["kind"] == "adapter"):
        raise typer.BadParameter("lora requires an adapter; base prompt comparisons require base weights")
    records = load_records(dataset / f"{split}.jsonl")
    if include_regressions:
        records += load_records(dataset / "regressions.jsonl")
    async def execute():
        backend = default_backend_factory(model)
        try:
            await backend.load(package, model)
            return await run_evaluation(backend, records, variant, GenerationConfig(max_tokens=max_tokens), output,
                {"artifact_id": model["artifact_id"], "base_model": model["base_model"],
                 "dataset_id": data["dataset_id"], "frozen_test_sha256": data["frozen_test_sha256"],
                 "split": split,
                 "selection_policy": "Test results must not select prompts or hyperparameters; operator compliance is not verified by this command."})
        finally:
            await backend.unload()
    report = asyncio.run(execute())
    emit({"classification": report["classification"], "quality": report["quality"], "output": output})


@app.command("benchmark")
def benchmark(url: str, model: str, prompts: Path, metadata: Path, output: Path,
              requests: int = 24, concurrency: int = 1, warmup: int = 2, max_tokens: int = 64,
              stream: bool = True, request_rate: float | None = None, repetition: int = 0,
              price_per_hour: float | None = None):
    from .benchmark import BenchmarkConfig, run_benchmark
    prompt_data = json.loads(prompts.read_text(encoding="utf-8"))
    engine_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    settings = BenchmarkConfig(requests=requests, concurrency=concurrency, warmup=warmup,
        max_tokens=max_tokens, stream=stream, request_rate=request_rate, repetition=repetition,
        compute_price_per_hour=price_per_hour)
    result = asyncio.run(run_benchmark(url, model, prompt_data, output, settings, engine_metadata,
                                      os.environ.get("LLM_API_KEY")))
    emit(result["summary"])


@app.command("compare-benchmarks")
def compare_benchmarks(reports: Annotated[list[Path], typer.Argument()], output: Path):
    from .benchmark import compare_reports
    comparison = compare_reports([json.loads(path.read_text(encoding="utf-8")) for path in reports])
    if output.exists():
        raise typer.BadParameter("Comparison output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    emit(comparison)


@review_app.command("prepare")
def prepare_review(dataset: Path, runs: Path, output: Path, sample_size: int = 12):
    """runs is a JSON object mapping opaque run labels to outputs.jsonl paths."""
    from .data import load_records, verify_dataset
    from .evaluation import make_blinded_review
    verify_dataset(dataset)
    mapping = json.loads(runs.read_text(encoding="utf-8"))
    outputs = {name: load_records(Path(path)) for name, path in mapping.items()}
    emit(make_blinded_review(load_records(dataset / "test.jsonl"), outputs, output, sample_size=sample_size))


@review_app.command("import")
def import_review(packet: Path, key: Path, output: Path):
    from .evaluation import import_human_review
    if output.exists():
        raise typer.BadParameter("Review output already exists")
    report = import_human_review(packet, key)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    emit(report)


@app.command("doctor")
def doctor():
    from .benchmark import environment
    emit(environment())


@app.command("audit-evidence")
def audit_evidence(root: Path = Path(".")):
    """Check local evidence integrity; external approval/publication remain separate."""
    from .evidence import validate_evidence
    result = validate_evidence(root)
    emit(result)
    if result["status"] == "fail":
        raise typer.Exit(code=1)
    if result["status"] == "pending":
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
