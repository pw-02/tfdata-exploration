import argparse
import time

import tensorflow as tf
import tensorflow_io as tfio  # important: registers s3:// filesystem support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatcher-address", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--worker-address",
        type=str,
        required=True,
        help="Advertised worker address host:port reachable by trainers.",
    )
    parser.add_argument(
        "--bind-address",
        type=str,
        default="0.0.0.0",
        help="Local interface to bind the worker server to.",
    )
    args = parser.parse_args()

    config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address=args.dispatcher_address,
        worker_address=args.worker_address,
        port=args.port,
        protocol="grpc",
        worker_tags=None,
    )

    server = tf.data.experimental.service.WorkerServer(config)

    print(
        f"Worker started: "
        f"bind_address={args.bind_address} "
        f"port={args.port} "
        f"dispatcher={args.dispatcher_address} "
        f"worker_address={args.worker_address}"
    )

    try:
        server.join()
    except KeyboardInterrupt:
        print(f"Stopping worker on port {args.port}...")


if __name__ == "__main__":
    main()