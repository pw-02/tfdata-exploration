import logging
import threading
import time

import tensorflow as tf
import tensorflow_io as tfio  # registers S3 filesystem support

from s3_utils import S3ImageDatasetBuilder

logging.basicConfig(level=logging.INFO)

import statistics


def run_trainer(name, trainer_id, builder, service, steps=20):
    ds, class_to_idx = builder.build_dataset(
        shuffle=True,
        service=service,
        job_name="shared-s3-image-demo",
        repeat=True,
        trainer_id=trainer_id,
        use_cross_trainer_cache=True,
    )

    it = iter(ds)
    t0 = time.perf_counter()

    step_fetch_times = []
    num_examples = 0
    first_batch_time = None

    for step in range(steps):
        s0 = time.perf_counter()
        images, labels = next(it)
        s1 = time.perf_counter()

        fetch_time = s1 - s0
        step_fetch_times.append(fetch_time)

        if step == 0:
            first_batch_time = s1 - t0
            print(f"{name}: first batch")
            print("  images:", images.shape, images.dtype)
            print("  labels:", labels.shape, labels.dtype)

        batch_size = int(images.shape[0])
        num_examples += batch_size

        print(f"{name}: step={step:03d} fetch_time={fetch_time:.4f}s")

    total_time = time.perf_counter() - t0
    avg_fetch = statistics.mean(step_fetch_times)
    p95_fetch = sorted(step_fetch_times)[int(0.95 * (len(step_fetch_times) - 1))]

    print(f"{name}: total_time={total_time:.4f}s")
    print(f"{name}: first_batch_time={first_batch_time:.4f}s")
    print(f"{name}: avg_fetch_time={avg_fetch:.4f}s")
    print(f"{name}: p95_fetch_time={p95_fetch:.4f}s")
    print(f"{name}: steps_per_sec={steps / total_time:.2f}")
    print(f"{name}: examples_per_sec={num_examples / total_time:.2f}")


def demo_with_tfdata_service_cross_trainer_cache():
    dispatcher = tf.data.experimental.service.DispatchServer(
        tf.data.experimental.service.DispatcherConfig(
            port=5000,
            work_dir="/tmp/tf_data_dispatcher",
        )
    )

    worker1 = tf.data.experimental.service.WorkerServer(
        tf.data.experimental.service.WorkerConfig(
            dispatcher_address="localhost:5000",
            port=5001,
        )
    )

    worker2 = tf.data.experimental.service.WorkerServer(
        tf.data.experimental.service.WorkerConfig(
            dispatcher_address="localhost:5000",
            port=5002,
        )
    )

    service = "grpc://localhost:5000"
    print("Dispatcher:", dispatcher.target)
    print("Service:", service)

    builder = S3ImageDatasetBuilder(
        path="s3://sdl-cifar10/train",
        image_size=224,
        batch_size=16,
    )

    t1 = threading.Thread(
        target=run_trainer,
        args=("TrainerA", "A", builder, service, 20),
    )
    t2 = threading.Thread(
        target=run_trainer,
        args=("TrainerB", "B", builder, service, 20),
    )

    t1.start()
    time.sleep(1.0)  # let TrainerA begin populating cache
    t2.start()

    t1.join()
    t2.join()


if __name__ == "__main__":
    demo_with_tfdata_service_cross_trainer_cache()