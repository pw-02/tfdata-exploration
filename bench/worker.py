import argparse

import tensorflow as tf
import tensorflow_io as tfio  # registers s3:// filesystem support


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
    args = parser.parse_args()

    config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address=args.dispatcher_address,
        worker_address=args.worker_address,
        port=args.port,
        protocol="grpc",
        data_transfer_protocol="grpc",
    )

    server = tf.data.experimental.service.WorkerServer(config)

    print(
        f"Worker started: port={args.port} "
        f"dispatcher={args.dispatcher_address} "
        f"worker_address={args.worker_address}"
    )

    server.join()


if __name__ == "__main__":
    main()