#!/usr/bin/env python3
"""Audit actual policy executions and compare two completed RoboTwin runs."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
POLICY = re.compile(r"^Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)", re.M)
FINAL = re.compile(r"Final success rate:\s*(\d+)/(\d+)")


def read_run(root, label):
    times = {}
    for line in (root / "task_times.tsv").read_text().splitlines():
        setting, task, gpu, started, finished, seconds, exit_code = line.split("\t")
        times[(setting, task)] = dict(seconds=int(seconds), started=started, finished=finished)
    configs, episodes = [], []
    paths = sorted((root / "logs").glob("eval_*.log"))
    assert len(paths) == 100, (label, len(paths))
    for path in paths:
        setting, task = path.stem.removeprefix("eval_").split("__", 1)
        text = ANSI.sub("", path.read_text(errors="replace")).replace("\r", "\n")
        matches = POLICY.findall(text)
        assert matches, path
        previous = 0
        seen_seeds = set()
        for index, (success_count, count, seed) in enumerate(matches, 1):
            success_count, count, seed = map(int, (success_count, count, seed))
            assert count == index, (path, count, index)
            assert success_count - previous in (0, 1), path
            assert seed not in seen_seeds, (path, seed)
            seen_seeds.add(seed)
            episodes.append(dict(run=label, setting=setting, task=task, seed=seed,
                                 success=success_count - previous, episode_index=index))
            previous = success_count
        printed_success, printed_total = map(int, FINAL.findall(text)[-1])
        assert previous == printed_success, path
        configs.append(dict(run=label, setting=setting, task=task, successes=previous,
                            attempts=len(matches), nominal_episodes=printed_total,
                            missing=printed_total - len(matches),
                            rate=previous / len(matches), policy_seeds=sorted(seen_seeds),
                            log=path.name, **times[(setting, task)]))
    return configs, episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-root", type=Path, required=True)
    parser.add_argument("--fp32-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = [("BF16_25", args.bf16_root), ("FP32_50", args.fp32_root)]
    configs, episodes, provenance = [], [], []
    for label, root in runs:
        config_rows, episode_rows = read_run(root, label)
        configs.extend(config_rows)
        episodes.extend(episode_rows)
        for path in sorted(root.rglob("*")):
            if path.is_file():
                provenance.append(dict(run=label, file=str(path.relative_to(root)),
                                       size=path.stat().st_size,
                                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    summaries = []
    for label, root in runs:
        for setting in ("demo_clean", "demo_randomized", "all"):
            rows = [r for r in configs if r["run"] == label and (setting == "all" or r["setting"] == setting)]
            success = sum(r["successes"] for r in rows)
            attempts = sum(r["attempts"] for r in rows)
            summaries.append(dict(run=label, setting=setting, configs=len(rows), successes=success,
                                  attempts=attempts, rate=success / attempts,
                                  missing=sum(r["missing"] for r in rows),
                                  incomplete_configs=sum(r["missing"] > 0 for r in rows),
                                  macro_rate=sum(r["rate"] for r in rows) / len(rows),
                                  task_seconds=sum(r["seconds"] for r in rows)))
    index = {(r["run"], r["setting"], r["task"]): r for r in configs}
    episode_index = {(r["run"], r["setting"], r["task"], r["seed"]): r for r in episodes}
    comparison = []
    for setting, task in sorted({(r["setting"], r["task"]) for r in configs}):
        b, f = index[("BF16_25", setting, task)], index[("FP32_50", setting, task)]
        common = sorted(set(b["policy_seeds"]) & set(f["policy_seeds"]))
        b_success = f_success = b_only = f_only = 0
        for seed in common:
            bv = episode_index[("BF16_25", setting, task, seed)]["success"]
            fv = episode_index[("FP32_50", setting, task, seed)]["success"]
            b_success += bv; f_success += fv
            b_only += int(bv > fv); f_only += int(fv > bv)
        comparison.append(dict(setting=setting, task=task, bf16=b, fp32=f,
                               difference=f["rate"] - b["rate"], common_seeds=len(common),
                               bf16_common_success=b_success, fp32_common_success=f_success,
                               bf16_only_success=b_only, fp32_only_success=f_only))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(summaries=summaries, comparison=comparison, episodes=episodes,
                  protocol_note="Same 10k weights, but BOTH inference precision and use_length changed. Expert/setup skips leave 38 incomplete configs per run. Rates are diagnostic, not a controlled precision ablation.")
    (args.output_dir / "eval_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.output_dir / "download_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    with (args.output_dir / "actual_task_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["setting", "task", "bf16_25_success", "bf16_25_actual_episodes", "bf16_25_rate", "fp32_50_success", "fp32_50_actual_episodes", "fp32_50_rate", "fp32_minus_bf16_pp", "bf16_missing", "fp32_missing", "bf16_minutes", "fp32_minutes"])
        for r in comparison:
            b, f = r["bf16"], r["fp32"]
            writer.writerow([r["setting"], r["task"], b["successes"], b["attempts"], b["rate"], f["successes"], f["attempts"], f["rate"], r["difference"] * 100, b["missing"], f["missing"], b["seconds"] / 60, f["seconds"] / 60])
    print(json.dumps(summaries, indent=2))
    for setting in ("demo_clean", "demo_randomized"):
        rows = [r for r in comparison if r["setting"] == setting]
        print(setting, "improved", sum(r["difference"] > 0 for r in rows), "worse", sum(r["difference"] < 0 for r in rows), "equal", sum(r["difference"] == 0 for r in rows))
        print("biggest decreases", [(r["task"], round(r["difference"] * 100, 1)) for r in sorted(rows, key=lambda r:r["difference"])[:6]])
        print("biggest increases", [(r["task"], round(r["difference"] * 100, 1)) for r in sorted(rows, key=lambda r:r["difference"], reverse=True)[:6]])


if __name__ == "__main__":
    main()
