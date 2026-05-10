from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List


RESULT_RE = re.compile(r"Result:\s*(True|False)")


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "actor"


def _unique_actor_names(paths: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    names: List[str] = []
    for path in paths:
        base = _safe_label(Path(path).stem)
        count = seen.get(base, 0)
        seen[base] = count + 1
        names.append(base if count == 0 else f"{base}_{count}")
    return names


def _resolved_scene(args: argparse.Namespace) -> tuple[int, int]:
    room_idx = int(args.room_idx if args.room_idx is not None else os.environ.get("ROOM_IDX", 1))
    table_idx = int(args.table_idx if args.table_idx is not None else os.environ.get("TABLE_IDX", 1))
    return room_idx, table_idx


def _resolved_model_path(args: argparse.Namespace) -> str:
    if getattr(args, "model_path", None):
        return str(Path(args.model_path).expanduser())
    return os.environ.get(
        "MODEL_PATH",
        str(Path.cwd() / "checkpoints" / "ego_vla_checkpoint" / "ckpt-human-video-pretrained"),
    )


def _resolve_actor_checkpoint_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_dir():
        preferred = [path / "actor.pt", path / "checkpoint.pt"]
        for candidate in preferred:
            if candidate.is_file():
                return candidate
        pt_files = sorted(path.glob("*.pt"))
        if len(pt_files) == 1:
            return pt_files[0]
        if not pt_files:
            raise FileNotFoundError(f"Actor checkpoint directory contains no .pt file: {path}")
        raise ValueError(
            f"Actor checkpoint directory contains multiple .pt files and no actor.pt: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Actor checkpoint path does not exist or is not a file/directory: {path}")
    return path


def _parse_results(path: Path) -> List[bool]:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval result file: {path}")
    results: List[bool] = []
    for line in path.read_text().splitlines():
        match = RESULT_RE.search(line)
        if match:
            results.append(match.group(1) == "True")
    if not results:
        raise ValueError(f"No 'Result: True/False' lines found in {path}")
    return results


def _run_eval(
    mode: str,
    args: argparse.Namespace,
    output_root: Path,
    actor_checkpoint: str | Path | None = None,
    run_name: str | None = None,
) -> Dict[str, object]:
    run_dir = output_root / (run_name or mode)
    result_path = run_dir / "results_local_eval.txt"
    room_idx, table_idx = _resolved_scene(args)
    model_path = _resolved_model_path(args)
    env = os.environ.copy()
    env["RL_MODE"] = mode
    env["RUN_DIR"] = str(run_dir)
    env["RESULT_PATH"] = str(result_path)
    env["NUM_EPISODES"] = str(args.num_episodes)
    env["NUM_TRIALS"] = str(args.num_trials)
    env["SAVE_VIDEO"] = "0" if args.no_save_video else "1"
    env.setdefault("SAVE_FRAMES", "0")
    env.setdefault("PROJECT_TRAJS", "0")
    if args.task:
        env["TASK"] = args.task
    env["MODEL_PATH"] = model_path
    env["ROOM_IDX"] = str(room_idx)
    env["TABLE_IDX"] = str(table_idx)
    if args.max_eval_steps is not None:
        env["MAX_EVAL_STEPS"] = str(args.max_eval_steps)
    if mode == "actor":
        if actor_checkpoint is None:
            raise ValueError("actor mode requires actor_checkpoint.")
        env["RL_ACTOR_CHECKPOINT"] = str(actor_checkpoint)
    else:
        env.pop("RL_ACTOR_CHECKPOINT", None)

    result_path.unlink(missing_ok=True)
    print(f"[paired-eval] running mode={mode} result_path={result_path}", flush=True)
    subprocess.run(["./run_local_eval.sh"], env=env, check=True)
    results = _parse_results(result_path)
    return {
        "mode": mode,
        "actor_checkpoint": None if actor_checkpoint is None else str(actor_checkpoint),
        "model_path": model_path,
        "room_idx": room_idx,
        "table_idx": table_idx,
        "run_dir": str(run_dir),
        "result_path": str(result_path),
        "results": results,
        "success_rate": sum(results) / len(results),
    }


def _run_paired_scene(args: argparse.Namespace, output_root: Path) -> Dict[str, object]:
    actor_checkpoints = [str(path) for path in args.actor_checkpoint]
    actor_names = _unique_actor_names(actor_checkpoints)

    runs = {"baseline": _run_eval("off", args, output_root)}
    if not args.skip_identity:
        runs["identity"] = _run_eval("identity", args, output_root)

    comparisons = {}
    actor_runs = {}
    for actor_name, actor_checkpoint in zip(actor_names, actor_checkpoints):
        run_name = "actor" if len(actor_checkpoints) == 1 else f"actor_{actor_name}"
        actor_run = _run_eval(
            "actor",
            args,
            output_root,
            actor_checkpoint=actor_checkpoint,
            run_name=run_name,
        )
        actor_runs[actor_name] = actor_run
        comparisons[actor_name] = _compare(
            runs["baseline"]["results"],
            None if args.skip_identity else runs["identity"]["results"],
            actor_run["results"],
        )

    if len(actor_checkpoints) == 1:
        runs["actor"] = next(iter(actor_runs.values()))
        comparison = next(iter(comparisons.values()))
        summary: Dict[str, object] = {"runs": runs, "comparison": comparison}
    else:
        runs["actors"] = actor_runs
        summary = {"runs": runs, "comparisons": comparisons}

    room_idx, table_idx = _resolved_scene(args)
    summary["scene"] = {"room_idx": room_idx, "table_idx": table_idx}
    summary["model_path"] = _resolved_model_path(args)
    return summary


def _actor_runs(summary: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    runs = summary["runs"]
    if "actors" in runs:
        return runs["actors"]
    return {"actor": runs["actor"]}


def _actor_comparisons(summary: Dict[str, object]) -> Dict[str, Dict[str, int | None]]:
    if "comparisons" in summary:
        return summary["comparisons"]
    return {"actor": summary["comparison"]}


def _aggregate_scene_summaries(scene_summaries: List[Dict[str, object]]) -> Dict[str, object]:
    if not scene_summaries:
        raise ValueError("No scene summaries to aggregate.")

    baseline_results: List[bool] = []
    first_summary = scene_summaries[0]["summary"]
    identity_results: List[bool] | None = [] if "identity" in first_summary["runs"] else None
    actor_results: Dict[str, List[bool]] = {name: [] for name in _actor_runs(first_summary).keys()}
    scenes = []

    for item in scene_summaries:
        summary = item["summary"]
        scene = summary["scene"]
        baseline = summary["runs"]["baseline"]["results"]
        baseline_results.extend(baseline)
        if identity_results is not None:
            identity_results.extend(summary["runs"]["identity"]["results"])
        for actor_name, actor_run in _actor_runs(summary).items():
            actor_results.setdefault(actor_name, []).extend(actor_run["results"])
        scenes.append(
            {
                "scene": scene,
                "summary_path": item["summary_path"],
                "baseline_success_rate": summary["runs"]["baseline"]["success_rate"],
                "actor_success_rates": {
                    actor_name: actor_run["success_rate"]
                    for actor_name, actor_run in _actor_runs(summary).items()
                },
                "comparisons": _actor_comparisons(summary),
            }
        )

    aggregate_actor_runs = {
        actor_name: {
            "results": results,
            "success_rate": sum(results) / len(results),
        }
        for actor_name, results in actor_results.items()
    }
    comparisons = {
        actor_name: _compare(baseline_results, identity_results, results)
        for actor_name, results in actor_results.items()
    }
    return {
        "format": "paired_eval_multi_scene_summary_v1",
        "model_path": scene_summaries[0]["summary"]["model_path"],
        "num_scenes": len(scene_summaries),
        "num_pairs": len(baseline_results),
        "scenes": scenes,
        "aggregate": {
            "baseline": {
                "results": baseline_results,
                "success_rate": sum(baseline_results) / len(baseline_results),
            },
            "identity": None
            if identity_results is None
            else {
                "results": identity_results,
                "success_rate": sum(identity_results) / len(identity_results),
            },
            "actors": aggregate_actor_runs,
            "comparisons": comparisons,
        },
    }


def _compare(baseline: List[bool], identity: List[bool] | None, actor: List[bool]) -> Dict[str, int | None]:
    if len(baseline) != len(actor):
        raise ValueError(
            "Paired eval result count mismatch: "
            f"baseline={len(baseline)} actor={len(actor)}"
        )
    if identity is not None and len(identity) != len(baseline):
        raise ValueError(
            "Paired eval result count mismatch: "
            f"baseline={len(baseline)} identity={len(identity)} actor={len(actor)}"
        )
    return {
        "num_pairs": len(baseline),
        "identity_matches_baseline": (
            None if identity is None else sum(b == i for b, i in zip(baseline, identity))
        ),
        "baseline_success_actor_fail": sum(b and not a for b, a in zip(baseline, actor)),
        "baseline_fail_actor_success": sum((not b) and a for b, a in zip(baseline, actor)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline/identity/actor paired eval with matched seeds.")
    parser.add_argument("--actor_checkpoint", required=True, nargs="+")
    parser.add_argument("--task", default=None)
    parser.add_argument(
        "--model_path",
        default=None,
        help="Frozen EgoVLA checkpoint path. Defaults to MODEL_PATH env, then run_local_eval.sh default.",
    )
    parser.add_argument("--room_idx", type=int, default=None)
    parser.add_argument("--table_idx", type=int, default=None)
    parser.add_argument(
        "--scene",
        nargs=2,
        type=int,
        action="append",
        metavar=("ROOM_IDX", "TABLE_IDX"),
        help="Evaluate one room/table scene. Can be repeated; when used, one top-level summary aggregates all scenes.",
    )
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--output_root", default=None)
    parser.add_argument(
        "--no_save_video",
        action="store_true",
        help="Disable paired eval mp4 recording. By default videos are saved.",
    )
    parser.add_argument(
        "--skip_identity",
        action="store_true",
        help="Run only baseline/off and actor. The summary sets identity_matches_baseline to null.",
    )
    args = parser.parse_args()

    actor_checkpoint_paths = [Path(path).expanduser() for path in args.actor_checkpoint]
    for path in actor_checkpoint_paths:
        _resolve_actor_checkpoint_path(path)
    args.actor_checkpoint = [str(path) for path in actor_checkpoint_paths]
    model_path = Path(_resolved_model_path(args)).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Frozen EgoVLA model path does not exist: {model_path}")
    args.model_path = str(model_path)

    output_root = Path(
        args.output_root
        or Path("playground_eval") / "paired_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print("Before running this wrapper, activate the simulator env:")
    print("  conda activate env_isaaclab")

    if args.scene:
        if args.room_idx is not None or args.table_idx is not None:
            raise ValueError("Use either repeated --scene or --room_idx/--table_idx, not both.")
        scene_summaries = []
        for room_idx, table_idx in args.scene:
            scene_args = argparse.Namespace(**vars(args))
            scene_args.room_idx = int(room_idx)
            scene_args.table_idx = int(table_idx)
            scene_root = output_root / f"room{room_idx}_table{table_idx}"
            scene_root.mkdir(parents=True, exist_ok=True)
            scene_summary = _run_paired_scene(scene_args, scene_root)
            scene_summary_path = scene_root / "paired_summary.json"
            scene_summary_path.write_text(json.dumps(scene_summary, indent=2))
            print(f"[paired-eval] scene_summary={scene_summary_path}")
            scene_summaries.append(
                {
                    "summary_path": str(scene_summary_path),
                    "summary": scene_summary,
                }
            )

        summary = _aggregate_scene_summaries(scene_summaries)
        summary_path = output_root / "paired_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[paired-eval] summary={summary_path}")
        print(json.dumps(summary, indent=2))
        return

    summary = _run_paired_scene(args, output_root)

    summary_path = output_root / "paired_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[paired-eval] summary={summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
