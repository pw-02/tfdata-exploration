import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def start_process(cmd, env=None):
    print("START:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def wait_seconds(seconds=3.0):
    time.sleep(seconds)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(metrics_list):
    print("\n=== SUMMARY ===")
    for m in metrics_list:
        input_fraction = (
            m["avg_fetch_time_sec"] / m["avg_step_time_sec"]
            if m["avg_step_time_sec"] > 0
            else 0.0
        )
        print(
            f"{m['trainer_name']:>10} | "
            f"mode={m['mode']:>13} | "
            f"model={m['model_name']:>12} | "
            f"first_batch={m['first_batch_time_sec']:.4f}s | "
            f"avg_fetch={m['avg_fetch_time_sec']:.4f}s | "
            f"avg_step={m['avg_step_time_sec']:.4f}s | "
            f"in_frac={input_fraction:.2f} | "
            f"p95_step={m['p95_step_time_sec']:.4f}s | "
            f"ex/sec={m['examples_per_sec']:.2f} | "
            f"loss={m['final_loss']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default="s3://sdl-cifar10/train")
    parser.add_argument("--model", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "resnet50"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--repeat", action="store_true")

    parser.add_argument("--mode", choices=["direct", "service", "service_cache"], default="service")
    parser.add_argument("--num-trainers", type=int, default=1)

    parser.add_argument(
        "--service",
        type=str,
        default=None,
        help="Use an existing tf.data service dispatcher, e.g. grpc://dispatcher-host:5000",
    )
    parser.add_argument(
        "--launch-local-service",
        default=True,
        action="store_true",
        help="Launch local dispatcher/workers instead of using --service.",
    )

    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--dispatcher-port", type=int, default=5000)
    parser.add_argument("--worker-base-port", type=int, default=5001)
    parser.add_argument("--job-name", type=str, default="image-benchmark")
    parser.add_argument("--trainer-stagger-sec", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default=".benchmark_results")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dispatcher = None
    workers = []
    trainers = []
    metrics_paths = []

    env = os.environ.copy()

    try:
        service = None

        if args.mode == "direct":
            service = None
        else:
            if args.launch_local_service:
                dispatcher = start_process([
                    sys.executable, "dispatcher.py",
                    "--port", str(args.dispatcher_port),
                    "--work-dir", "/tmp/tf_data_dispatcher_bench",
                ])
                wait_seconds(2.0)

                for i in range(args.num_workers):
                    port = args.worker_base_port + i
                    worker_address = f"localhost:{port}"
                    p = start_process([
                        sys.executable, "worker.py",
                        "--dispatcher-address", f"localhost:{args.dispatcher_port}",
                        "--port", str(port),
                        "--worker-address", worker_address,
                    ])
                    workers.append(p)

                wait_seconds(3.0)
                service = f"grpc://localhost:{args.dispatcher_port}"
            else:
                if not args.service:
                    raise ValueError(
                        "For mode=service or mode=service_cache, provide either "
                        "--service grpc://host:port or --launch-local-service"
                    )
                service = args.service

        for i in range(args.num_trainers):
            trainer_name = f"Trainer{i}"
            trainer_id = f"T{i}"
            metrics_path = out_dir / f"{trainer_name}_{args.mode}_{args.model}.json"
            metrics_paths.append(str(metrics_path))

            cmd = [
                sys.executable, "trainer.py",
                "--trainer-name", trainer_name,
                "--trainer-id", trainer_id,
                "--mode", args.mode,
                "--model", args.model,
                "--path", args.path,
                "--image-size", str(args.image_size),
                "--batch-size", str(args.batch_size),
                "--steps", str(args.steps),
                "--learning-rate", str(args.learning_rate),
                "--job-name", args.job_name,
                "--metrics-out", str(metrics_path),
            ]

            if args.shuffle:
                cmd.append("--shuffle")
            if args.repeat:
                cmd.append("--repeat")
            if service is not None:
                cmd.extend(["--service", service])

            p = start_process(cmd, env=env)
            trainers.append(p)

            if i < args.num_trainers - 1:
                time.sleep(args.trainer_stagger_sec)

        for p in trainers:
            rc = p.wait()
            # if rc != 0:
            #     raise RuntimeError(f"Trainer failed with rc={rc}")

        metrics = [load_json(path) for path in metrics_paths]
        summarize(metrics)

        summary_path = out_dir / f"summary_{args.mode}_{args.model}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print(f"\nWrote summary to {summary_path}")

    finally:
        for p in trainers:
            if p.poll() is None:
                p.terminate()
        for p in workers:
            if p.poll() is None:
                p.terminate()
        if dispatcher is not None and dispatcher.poll() is None:
            dispatcher.terminate()


if __name__ == "__main__":
    main()