import argparse
import socket
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
        default=None,
        help="Advertised worker address host:port.",
    )
    args = parser.parse_args()

    worker_address = args.worker_address
    if worker_address is None:
        hostname = socket.gethostname()
        worker_address = f"{hostname}:{args.port}"

    server = tf.data.experimental.service.WorkerServer(
        tf.data.experimental.service.WorkerConfig(
            dispatcher_address=args.dispatcher_address,
            worker_address=worker_address,
            port=args.port,
        )
    )

    print(
        f"Worker started: port={args.port} "
        f"dispatcher={args.dispatcher_address} "
        f"worker_address={worker_address}"
    )

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print(f"Stopping worker on port {args.port}...")


if __name__ == "__main__":
    main()