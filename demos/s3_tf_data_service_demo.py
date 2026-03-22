import logging
import tensorflow as tf
import tensorflow_io as tfio  # often needed for s3:// filesystem support

from s3_utils import S3ImageDatasetBuilder

logging.basicConfig(level=logging.INFO)


def demo_with_tfdata_service():
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
        path="s3://sdl-cifar10/train", #"s3://imagenet1k-sdl/train/",
        image_size=224,
        batch_size=16,
    )

    ds, class_to_idx = builder.build_dataset(
        shuffle=True,
        service=service,
        job_name="s3-image-demo",
    )

    print("class_to_idx:", class_to_idx)

    for i, (images, labels) in enumerate(ds.take(2)):
        print(f"Batch {i}")
        print("  images:", images.shape, images.dtype)
        print("  labels:", labels.shape, labels.dtype)


if __name__ == "__main__":
    demo_with_tfdata_service()