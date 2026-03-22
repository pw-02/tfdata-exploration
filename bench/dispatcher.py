import argparse
import os
import time

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--work-dir", type=str, default="/tmp/tf_data_dispatcher")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    server = tf.data.experimental.service.DispatchServer(
        tf.data.experimental.service.DispatcherConfig(
            port=args.port,
            work_dir=args.work_dir,
        )
    )

    print(f"Dispatcher target: {server.target}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping dispatcher...")


if __name__ == "__main__":
    main()