import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, cwd=None):
    print("\nRUN:", " ".join(cmd))
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=cwd)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with rc={result.returncode}: {' '.join(cmd)}")
    return elapsed


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_run_summary(run_config, trainer_metrics_list, wall_time_sec):
    rows = []
    for m in trainer_metrics_list:
        row = {
            "run_name": run_config["run_name"],
            "mode": run_config["mode"],
            "model": run_config["model"],
            "path": run_config["path"],
            "image_size": run_config["image_size"],
            "batch_size": run_config["batch_size"],
            "steps": run_config["steps"],
            "learning_rate": run_config["learning_rate"],
            "shuffle": run_config["shuffle"],
            "repeat": run_config["repeat"],
            "num_trainers": run_config["num_trainers"],
            "num_workers": run_config["num_workers"],
            "trainer_stagger_sec": run_config["trainer_stagger_sec"],
            "job_name": run_config["job_name"],
            "service": run_config["service"],
            "launch_local_service": run_config["launch_local_service"],
            "wall_time_sec": wall_time_sec,
            **m,
        }
        row["input_fraction"] = (
            (m["avg_fetch_time_sec"] / m["avg_step_time_sec"])
            if m["avg_step_time_sec"] > 0
            else 0.0
        )
        rows.append(row)
    return rows


def write_csv(rows, out_path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\n=== MATRIX SUMMARY ===")
    for row in rows:
        print(
            f"{row['run_name']:<40} "
            f"{row['trainer_name']:<10} "
            f"mode={row['mode']:<13} "
            f"fetch={row['avg_fetch_time_sec']:.4f}s "
            f"step={row['avg_step_time_sec']:.4f}s "
            f"in_frac={row['input_fraction']:.2f} "
            f"ex/sec={row['examples_per_sec']:.2f}"
        )


def make_run_name(cfg):
    parts = [
        cfg["mode"],
        cfg["model"],
        f"bs{cfg['batch_size']}",
        f"steps{cfg['steps']}",
        f"trainers{cfg['num_trainers']}",
    ]
    if cfg["mode"] in {"service", "service_cache"}:
        parts.append(f"workers{cfg['num_workers']}")
    parts.append("shuffle" if cfg["shuffle"] else "noshuffle")
    parts.append("repeat" if cfg["repeat"] else "norepeat")
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default="s3://sdl-cifar10/train")

    parser.add_argument(
        "--modes",
        nargs="+",
        default=["direct", "service", "service_cache"],
        choices=["direct", "service", "service_cache"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mobilenet_v2"],
        choices=["mobilenet_v2", "resnet50"],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--steps-list", nargs="+", type=int, default=[100])
    parser.add_argument("--num-trainers-list", nargs="+", type=int, default=[1])
    parser.add_argument("--num-workers-list", nargs="+", type=int, default=[1,2])

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    parser.add_argument("--shuffle-options", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--repeat-options", nargs="+", type=int, default=[1])

    parser.add_argument("--trainer-stagger-sec", type=float, default=1.0)
    parser.add_argument("--job-name-prefix", type=str, default="image-benchmark")
    parser.add_argument("--results-root", type=str, default="./bench/bmatrix_results")

    parser.add_argument("--service", type=str, default=None)
    parser.add_argument("--launch-local-service", action="store_true", default=True)
    parser.add_argument("--dispatcher-port", type=int, default=5000)
    parser.add_argument("--worker-base-port", type=int, default=5001)

    parser.add_argument("--run-benchmark-script", type=str, default="bench/run_benchmark.py")
    parser.add_argument("--cwd", type=str, default=".")

    args = parser.parse_args()

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    manifest = []
    experiment_index = 0

    for mode, model, batch_size, steps, num_trainers, shuffle_i, repeat_i in itertools.product(
        args.modes,
        args.models,
        args.batch_sizes,
        args.steps_list,
        args.num_trainers_list,
        args.shuffle_options,
        args.repeat_options,
    ):
        shuffle = bool(shuffle_i)
        repeat = bool(repeat_i)

        worker_values = args.num_workers_list if mode in {"service", "service_cache"} else [0]

        for num_workers in worker_values:
            run_cfg = {
                "mode": mode,
                "model": model,
                "path": args.path,
                "image_size": args.image_size,
                "batch_size": batch_size,
                "steps": steps,
                "learning_rate": args.learning_rate,
                "shuffle": shuffle,
                "repeat": repeat,
                "num_trainers": num_trainers,
                "num_workers": num_workers,
                "trainer_stagger_sec": args.trainer_stagger_sec,
                "job_name": f"{args.job_name_prefix}-{experiment_index}",
                "service": args.service,
                "launch_local_service": args.launch_local_service,
            }
            run_cfg["run_name"] = make_run_name(run_cfg)

            out_dir = results_root / run_cfg["run_name"]
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                args.run_benchmark_script,
                "--mode", run_cfg["mode"],
                "--model", run_cfg["model"],
                "--path", run_cfg["path"],
                "--image-size", str(run_cfg["image_size"]),
                "--batch-size", str(run_cfg["batch_size"]),
                "--steps", str(run_cfg["steps"]),
                "--learning-rate", str(run_cfg["learning_rate"]),
                "--num-trainers", str(run_cfg["num_trainers"]),
                "--trainer-stagger-sec", str(run_cfg["trainer_stagger_sec"]),
                "--job-name", run_cfg["job_name"],
                "--out-dir", str(out_dir),
            ]

            if run_cfg["mode"] in {"service", "service_cache"}:
                if run_cfg["launch_local_service"]:
                    cmd.append("--launch-local-service")
                    cmd.extend([
                        "--num-workers", str(run_cfg["num_workers"]),
                        "--dispatcher-port", str(args.dispatcher_port),
                        "--worker-base-port", str(args.worker_base_port),
                    ])
                else:
                    if not run_cfg["service"]:
                        raise ValueError(
                            "For service/service_cache matrix runs, provide either "
                            "--service grpc://host:port or --launch-local-service"
                        )
                    cmd.extend(["--service", run_cfg["service"]])

            if run_cfg["shuffle"]:
                cmd.append("--shuffle")
            if run_cfg["repeat"]:
                cmd.append("--repeat")

            wall_time_sec = run_command(cmd, cwd=args.cwd)

            summary_path = out_dir / f"summary_{run_cfg['mode']}_{run_cfg['model']}.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Expected summary file not found: {summary_path}")

            trainer_metrics_list = load_json(summary_path)
            rows = flatten_run_summary(run_cfg, trainer_metrics_list, wall_time_sec)
            all_rows.extend(rows)

            manifest.append({
                "run_config": run_cfg,
                "summary_path": str(summary_path),
                "wall_time_sec": wall_time_sec,
            })

            experiment_index += 1

    print_summary(all_rows)

    json_path = results_root / "matrix_results.json"
    csv_path = results_root / "matrix_results.csv"
    manifest_path = results_root / "matrix_manifest.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, sort_keys=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    write_csv(all_rows, csv_path)

    print(f"\nWrote JSON: {json_path}")
    print(f"Wrote CSV:  {csv_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()