import argparse
import json

from benchmark_lib import run_trainer_benchmark, write_metrics_json


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--trainer-name", type=str, required=True)
    parser.add_argument("--trainer-id", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["direct", "service", "service_cache"], required=True)

    parser.add_argument("--model", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "resnet50"])

    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--repeat", action="store_true")

    parser.add_argument("--service", type=str, default=None)
    parser.add_argument("--job-name", type=str, default="image-benchmark")
    parser.add_argument("--metrics-out", type=str, required=True)

    args = parser.parse_args()

    use_cross_trainer_cache = args.mode == "service_cache"
    service = None if args.mode == "direct" else args.service

    if args.mode != "direct" and not service:
        raise ValueError("--service must be provided for mode=service or mode=service_cache")

    metrics = run_trainer_benchmark(
        trainer_name=args.trainer_name,
        trainer_id=args.trainer_id,
        mode=args.mode,
        model_name=args.model,
        path=args.path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        steps=args.steps,
        shuffle=args.shuffle,
        service=service,
        job_name=args.job_name,
        repeat=args.repeat,
        use_cross_trainer_cache=use_cross_trainer_cache,
        learning_rate=args.learning_rate,
    )

    print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
    write_metrics_json(metrics, args.metrics_out)


if __name__ == "__main__":
    main()