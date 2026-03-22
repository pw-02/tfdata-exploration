import tensorflow as tf
import tensorflow_io as tfio  # often needed for s3:// filesystem support
from s3_utils import S3ImageDatasetBuilder
import logging

logging.basicConfig(level=logging.INFO)

def demo_plain_tfdata():
    logging.basicConfig(level=logging.INFO)
    builder = S3ImageDatasetBuilder(
        path="s3://sdl-cifar10/train", #"s3://imagenet1k-sdl/train/",
        image_size=224,
        batch_size=16,
    )

    ds, class_to_idx = builder.build_dataset(shuffle=True, service=None)

    print("class_to_idx:", class_to_idx)
    for i, (images, labels) in enumerate(ds.take(2)):
        print(f"Batch {i}")
        print("  images:", images.shape, images.dtype)
        print("  labels:", labels.shape, labels.dtype)


if __name__ == "__main__":
    demo_plain_tfdata()