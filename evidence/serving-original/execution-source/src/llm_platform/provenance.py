"""Preserve source bytes independently of later working-tree edits."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


def capture_source_snapshot(root: Path, destination: Path, revision: str | None = None) -> dict:
    """Capture current source, or explicitly reconstruct an immutable historical tree.

    Historical reconstruction is labeled retrospective. It cannot claim a file
    was captured at process startup. Current snapshots require committed source
    and preserve raw filesystem bytes, including their actual line endings.
    """
    root, destination = Path(root).resolve(), Path(destination)
    if destination.exists():
        raise FileExistsError("Source snapshot must use a new destination")
    historical = revision is not None
    if revision is None:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "scripts"],
                                        cwd=root, text=True).strip()
        if dirty:
            raise ValueError("Commit source and experiment scripts before capturing execution provenance")
    if not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", revision):
        raise ValueError("Source snapshot requires a complete immutable revision")
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", revision, "--", "src", "scripts"], cwd=root)
    entries = [entry.decode("utf-8").split("\t", 1) for entry in tree.split(b"\0") if entry]
    files = {}
    for description, name in sorted(entries, key=lambda entry: entry[1]):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe source path")
        if path.suffix != ".py":
            continue
        if description.split()[0] not in {"100644", "100755"}:
            raise ValueError("Source snapshot requires ordinary committed source files, not links")
        source = root / name
        if not historical and (source.is_symlink() or not source.resolve().is_relative_to(root)):
            raise ValueError("Source snapshot cannot follow links outside the project")
        content = (subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root)
                   if historical else source.read_bytes())
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files[name] = hashlib.sha256(content).hexdigest()
    if not files:
        raise ValueError("Source tree has no project Python files")
    manifest = {"schema_version": 1, "source_revision": revision,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "mode": ("retrospective exact Git tree reconstruction" if historical else "pre-generation filesystem source capture"),
        "files": files,
        "limitations": ["A source inventory establishes bytes and version identity, not remote execution attestation."]}
    manifest["snapshot_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
