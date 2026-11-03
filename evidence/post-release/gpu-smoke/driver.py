"""Real post-fix GPU HTTP smoke; this is not a throughput or quality benchmark."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'scripts'))
import run_serving_experiments as suite
from llm_platform.provenance import capture_source_snapshot
from llm_platform.registry import verify_package
from llm_platform.benchmark import _request


async def execute():
    output = root / 'evidence/post-release/gpu-smoke'
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / 'driver.py')
    snapshot = capture_source_snapshot(root, output / 'execution-source')
    base = root / 'runtime/quality-002/base-package'
    adapter = root / 'evidence/adapter-package'
    vllm_adapter = root / 'runtime/serving-001/packages/adapter-vllm'
    packages = [base, adapter, vllm_adapter]
    manifests = {path: verify_package(path) for path in packages}
    assert suite.semantic_ids(manifests[adapter]) == suite.semantic_ids(manifests[vllm_adapter])
    results = []
    inventory = {
        'transformers': suite.software_inventory(sys.executable),
        'vllm': suite.software_inventory('/home/ridle/llm-platform-vllm-env/bin/python')}
    for engine, package in [('transformers', adapter), ('vllm', vllm_adapter)]:
        manifest = manifests[package]
        async with suite.server_session(engine, package, manifest, packages,
                root / f'runtime/post-release-gpu-smoke/{engine}', output / engine,
                Path('/home/ridle/llm-platform-vllm-env/bin/python'), inventory[engine]) as session:
            headers = {'Authorization': 'Bearer ' + session['api_key']}
            async with suite.httpx.AsyncClient(base_url=session['url'], headers=headers, timeout=90) as client:
                ready = await client.get('/health/ready')
                assert ready.status_code == 200 and ready.json()['model'] == manifest['artifact_id']
                sync = await suite.completion(client, manifest['artifact_id'], 8)
                assert sync['choices'][0]['finish_reason'] in {'stop', 'length'}
                assert sync['choices'][0]['message']['content'].strip()
                config = suite.BenchmarkConfig(requests=1, max_tokens=8, warmup=0)
                streamed = await _request(
                    client, manifest['artifact_id'], suite.SHORT_PROMPTS[0], config, 0)
                assert streamed['success'] and streamed['output_tokens'] > 0
            admin_headers = {'Authorization': 'Bearer ' + session['admin_key']}
            async with suite.httpx.AsyncClient(base_url=session['url'], headers=admin_headers, timeout=10) as admin:
                state = await suite.wait_idle(admin)
                assert state['active_model'] == manifest['artifact_id']
                assert state['engine']['active_requests'] == 0
            results.append({'engine': engine, 'artifact_id': manifest['artifact_id'],
                            'ready_status': ready.status_code, 'sync': sync, 'streamed': streamed,
                            'ready_engine': session['startup']['ready_engine_state']})
    record = {'status': 'pass', 'scope': 'Real RTX 4060 CUDA adapter loading and synchronous/SSE HTTP smoke after code fixes; no new quality or comparative performance claim.',
              'source_revision': snapshot['source_revision'],
              'execution_source_snapshot_sha256': snapshot['snapshot_sha256'],
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gpu_inventory': suite.environment()['gpu_inventory_csv'], 'results': results}
    suite.write_json(output / 'report.json', record)
    print(json.dumps({'status': record['status'], 'engines': [row['engine'] for row in results]}, indent=2))


if __name__ == '__main__':
    os.chdir(root)
    asyncio.run(execute())
