#!/usr/bin/env python3
"""Recompute cross-host evidence from four archived rank-0 traces, without a GPU.

Only the requested JSON output is written. Input reports and traces are read-only.
CPU duration totals are inclusive sums, not wall time: nested CPU events overlap.
"""

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics


MA11_OLD = (
    "TempName/logs/h100/phase6/"
    "ma11-v1-causal-softmax-1x-50steps-20261027-202038-900556/fullstep"
)
MA11_NEW = (
    "Load/logs/ma14-v2-epoch-reuse-1x-50steps-20261031-030311-454459/fullstep"
)
BASIC_OLD = (
    "TempName/logs/h100/phase6/"
    "basic-quant-receiver-1x-50steps-20261025-090410-798870/compare1"
)
BASIC_NEW = (
    "Load/logs/mav2-then-ma14-ma18-20261031-014137-056456/mav2/compare1"
)
RUNS = {
    "ma11_old": (MA11_OLD, "F4-cache-zcount-csoftmax"),
    "ma11_new": (MA11_NEW, "F4-cache-zcount-csoftmax"),
    "basic_old": (BASIC_OLD, "basic"),
    "basic_new": (BASIC_NEW, "basic"),
}
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def duration_stats(events):
    durations = [event["dur"] for event in events]
    return {
        "count": len(durations),
        "inclusive_duration_sum_ms": sum(durations) / 1000,
        "median_duration_us": statistics.median(durations) if durations else None,
    }


def union_stats(events):
    intervals = sorted((event["ts"], event["ts"] + event["dur"]) for event in events)
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    if not merged:
        return None
    span = merged[-1][1] - merged[0][0]
    busy = sum(end - start for start, end in merged)
    gaps = [merged[index][0] - merged[index - 1][1] for index in range(1, len(merged))]
    return {
        "interval_count": len(intervals),
        "merged_interval_count": len(merged),
        "span_ms": span / 1000,
        "busy_union_ms": busy / 1000,
        "gap_ms": (span - busy) / 1000,
        "gap_count": len(gaps),
        "median_gap_us": statistics.median(gaps) if gaps else None,
        "gaps_above_10us_count": sum(gap > 10 for gap in gaps),
        "gaps_above_10us_sum_ms": sum(gap for gap in gaps if gap > 10) / 1000,
    }


def summarize_trace(workspace, relative_path):
    path = workspace / relative_path
    with gzip.open(path, "rt") as source:
        trace = json.load(source)
    events = [
        event for event in trace["traceEvents"]
        if event.get("ph") == "X" and "dur" in event
    ]
    by_category = collections.defaultdict(list)
    by_category_name = collections.defaultdict(lambda: collections.defaultdict(list))
    by_gpu = collections.defaultdict(list)
    for event in events:
        category = event.get("cat", "")
        by_category[category].append(event)
        by_category_name[category][event["name"]].append(event)
        if category in GPU_CATEGORIES:
            by_gpu[str(event["pid"])].append(event)
    trace_spans = by_category.get("Trace", [])
    return {
        "trace_path": relative_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "all_event_count": len(trace["traceEvents"]),
        "complete_duration_event_count": len(events),
        "metadata": {
            "schemaVersion": trace.get("schemaVersion"),
            "cupti_version": trace.get("cupti_version"),
            "cuda_runtime_api_version": trace.get("cuda_runtime_version"),
            "cuda_driver_api_version": trace.get("cuda_driver_version"),
            "driver_version_note": "CUDA driver API compatibility number, not installed driver package version",
            "deviceProperties": trace.get("deviceProperties"),
            "distributedInfo": trace.get("distributedInfo"),
        },
        "trace_spans": [
            {"name": event["name"], "duration_ms": event["dur"] / 1000}
            for event in trace_spans
        ],
        "categories": {
            category: duration_stats(group)
            for category, group in sorted(by_category.items())
        },
        "by_category_and_name": {
            category: {
                name: duration_stats(group)
                for name, group in sorted(names.items())
            }
            for category, names in sorted(by_category_name.items())
        },
        "gpu_activity": {
            "categories": sorted(GPU_CATEGORIES),
            "method": "Merge overlapping intervals per GPU pid; span begins at first GPU event and ends at last GPU event.",
            "by_gpu_pid": {
                pid: union_stats(group) for pid, group in sorted(by_gpu.items())
            },
        },
    }


def read_report(workspace, directory, arm):
    report_path = directory + "/report.json"
    report = json.loads((workspace / report_path).read_text())
    if arm == "basic":
        combined = report["rounds"]["combined"]
        samples = combined["ms_per_step"][arm]
        timing = report["execution_timing"]
        model = report["config"]
        identity = report["measurement_identity"]["fingerprints"]
    else:
        samples = [
            sample["ms_per_step"] for sample in report["all_samples"]
            if sample["arm"] == arm
        ]
        plan = json.loads((workspace / directory / "plan.json").read_text())
        timing = {key: plan[key] for key in ("warmup_steps", "timed_steps")}
        model = report["model"]
        identity = report["rank_identities"]
    result = {
        "report_path": report_path,
        "arm": arm,
        "mean_ms_per_step": statistics.mean(samples),
        "median_window_ms_per_step": statistics.median(samples),
        "all_window_ms_per_step": samples,
        "arm_window_count": len(samples),
        "execution_timing": timing,
        "model": model,
        "rank_identities_or_fingerprints": identity,
        "report_correctness_passed": report.get("correctness_passed"),
    }
    if arm == "basic":
        result["mean_loss"] = combined["mean_loss"][arm]
        result["mean_final_loss"] = combined["mean_final_loss"][arm]
        result["correctness_note"] = (
            "The new MAV2 report fails candidate quality gates; that is a separate issue from basic arm slowdown."
        )
    return result


def compare_formal_metrics(formal):
    result = {}
    for family in ("ma11", "basic"):
        old = formal[family + "_old"]["mean_ms_per_step"]
        new = formal[family + "_new"]["mean_ms_per_step"]
        result[family] = {
            "old_ms_per_step": old,
            "new_ms_per_step": new,
            "increase_ms_per_step": new - old,
            "time_increase_percent": (new / old - 1) * 100,
            "throughput_decrease_percent": (1 - old / new) * 100,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, help="Write JSON here; otherwise print JSON to stdout")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    formal = {
        label: read_report(workspace, directory, arm)
        for label, (directory, arm) in RUNS.items()
    }
    result = {
        "method": {
            "scope": "Four archived rank-0 traces and their formal timing reports; no new GPU experiments",
            "trace_units": "Chrome trace ts/dur are microseconds; reported totals/spans are milliseconds",
            "cpu_duration_caution": "Inclusive sums of nested cpu_op/user_annotation events overlap. Never add these sums as wall-clock time.",
            "gpu_duration_caution": "GPU kernel duration sums can overlap and include communication waiting. busy_union_ms merges intervals, but does not distinguish useful computation from waiting.",
            "cross_host_caution": "Profile has instrumentation overhead. Its durations diagnose mechanism and cannot be directly equated to unprofiled step losses. Hardware, driver, and runtime host differences remain confounded.",
            "basic_baseline_choice": "Use basic-quant-receiver v2; a-baseline-fusion v1 uses a different backend and numerical trajectory.",
        },
        "formal_metrics": formal,
        "formal_comparisons": compare_formal_metrics(formal),
        "traces": {
            label: summarize_trace(workspace, directory + "/results/profile-" + arm + "/rank0-profile.json.gz")
            for label, (directory, arm) in RUNS.items()
        },
    }
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
