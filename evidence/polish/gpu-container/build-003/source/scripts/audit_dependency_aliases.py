"""Deduplicate advisory identities without treating matches as proven exploits.

Inputs are saved pip-audit JSON results. IDs and aliases form a transitive union;
raw occurrences, primary IDs, affected package versions, and skips remain visible.
This performs no network access and never imports or executes audited packages.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def deduplicate(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    parent: dict[str, str] = {}
    records = []

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = sorted((find(left), find(right)))
        parent[b] = a

    for inventory, report in reports.items():
        if not isinstance(report, dict) or not isinstance(report.get("dependencies"), list):
            raise ValueError("Each input must contain a pip-audit dependencies list")
        for dependency in report["dependencies"]:
            name = re.sub(r"[-_.]+", "-", dependency["name"].lower())
            for advisory in dependency.get("vulns", []):
                identity = advisory.get("id")
                aliases = advisory.get("aliases", [])
                if (not isinstance(identity, str) or not identity.strip() or not isinstance(aliases, list)
                        or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)):
                    raise ValueError("Advisory IDs and aliases must be nonempty strings")
                identities = sorted(set([identity, *aliases]))
                for alias in identities:
                    union(identity, alias)
                records.append({"inventory": inventory, "package": name, "version": dependency.get("version"),
                                "primary_id": identity, "identities": identities,
                                "fix_versions": advisory.get("fix_versions", []),
                                "description": advisory.get("description", "")})

    groups = defaultdict(list)
    for record in records:
        groups[find(record["primary_id"])].append(record)
    clusters = []
    for representative, members in sorted(groups.items()):
        clusters.append({"cluster_id": representative,
                         "identities": sorted({identity for member in members for identity in member["identities"]}),
                         "raw_occurrences": len(members), "records": members,
                         "exploitability": "Not determined by advisory matching or alias deduplication"})
    inventory_results = {}
    for inventory, report in reports.items():
        members = [record for record in records if record["inventory"] == inventory]
        packages = {}
        for name in sorted({record["package"] for record in members}):
            package_records = [record for record in members if record["package"] == name]
            packages[name] = {"versions": sorted({record["version"] for record in package_records}),
                              "raw_occurrences": len(package_records),
                              "alias_deduplicated_advisories": len({find(record["primary_id"]) for record in package_records})}
        inventory_results[inventory] = {
            "dependencies_queried": len(report["dependencies"]), "raw_occurrences": len(members),
            "distinct_primary_ids": len({record["primary_id"] for record in members}),
            "alias_deduplicated_advisories": len({find(record["primary_id"]) for record in members}),
            "affected_packages": packages,
            "skipped": [{"name": dependency["name"], "reason": dependency["skip_reason"]}
                        for dependency in report["dependencies"] if dependency.get("skip_reason")],
        }
    return {"schema_version": 1, "method": "Transitive union of every primary ID and alias across all supplied inventories",
            "scope": "Counts are advisory-database matches, not demonstrated exploitable platform defects. Skips are not clean results.",
            "raw_occurrences": len(records), "alias_deduplicated_advisories": len(clusters),
            "inventories": inventory_results, "clusters": clusters}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="Unique LABEL=JSON_PATH; repeat for each inventory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output path to preserve prior audit evidence")
    reports, sources = {}, {}
    for specification in args.source:
        label, separator, filename = specification.partition("=")
        if not separator or not label or label in reports:
            raise ValueError("Each source requires a unique nonempty LABEL=JSON_PATH")
        path = Path(filename)
        raw = path.read_bytes()
        reports[label] = json.loads(raw)
        sources[label] = {"path": path.as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}
    result = deduplicate(reports)
    result.update(created_at_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"alias_deduplicated_advisories": result["alias_deduplicated_advisories"],
                      "inventories": result["inventories"]}, indent=2))


if __name__ == "__main__":
    main()
