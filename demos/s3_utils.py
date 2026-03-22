import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import boto3
import botocore
import tensorflow as tf
import tensorflow_io as tfio  # often needed for s3:// support


@dataclass
class S3Url:
    bucket: str
    key: str

    @classmethod
    def parse(cls, url: str) -> "S3Url":
        if not url.startswith("s3://"):
            raise ValueError(f"Expected s3:// URL, got: {url}")
        no_scheme = url[len("s3://"):]
        parts = no_scheme.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return cls(bucket=bucket, key=key)


class S3ImageDatasetBuilder:
    def __init__(
        self,
        path: str,
        image_size: int = 224,
        batch_size: int = 32,
        logger: Optional[logging.Logger] = None,
    ):
        self.path = path.rstrip("/") + "/"
        self.image_size = image_size
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)
        self.s3 = boto3.client("s3")

    def get_samples_from_s3(self) -> Dict[str, List[str]]:
        """
        Returns:
            {
                "cat": ["train/cat/a.jpg", "train/cat/b.jpg"],
                "dog": ["train/dog/c.jpg"]
            }
        """
        s3_url = S3Url.parse(self.path)
        s3_bucket = s3_url.bucket
        s3_prefix = s3_url.key.rstrip("/")
        index_key = f"{s3_prefix}/samples.json"

        try:
            obj = self.s3.get_object(Bucket=s3_bucket, Key=index_key)
            self.logger.info("Loaded cached S3 index from s3://%s/%s", s3_bucket, index_key)
            return json.loads(obj["Body"].read().decode("utf-8"))
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code not in {"NoSuchKey", "404"}:
                raise
            self.logger.warning(
                "No existing S3 index found at s3://%s/%s, scanning bucket.",
                s3_bucket,
                index_key,
            )

        samples: Dict[str, List[str]] = {}
        paginator = self.s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix + "/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                rel = key[len(s3_prefix):].lstrip("/")
                parts = rel.split("/")
                if len(parts) < 2:
                    # Skip files not under class_name/file structure
                    continue

                class_name = parts[0]
                samples.setdefault(class_name, []).append(key)

        if not samples:
            raise RuntimeError(f"No image files found under {self.path}")

        self.s3.put_object(
            Bucket=s3_bucket,
            Key=index_key,
            Body=json.dumps(samples).encode("utf-8"),
            ContentType="application/json",
        )
        self.logger.info("Created S3 index at s3://%s/%s", s3_bucket, index_key)

        return samples

    def flatten_samples(
        self, samples: Dict[str, List[str]]
    ) -> Tuple[List[str], List[int], Dict[str, int]]:
        """
        Converts:
            {"cat": ["train/cat/a.jpg"], "dog": ["train/dog/b.jpg"]}
        into:
            paths, labels, class_to_idx
        """
        s3_url = S3Url.parse(self.path)
        bucket = s3_url.bucket

        class_names = sorted(samples.keys())
        class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

        paths: List[str] = []
        labels: List[int] = []

        for class_name in class_names:
            for key in samples[class_name]:
                paths.append(f"s3://{bucket}/{key}")
                labels.append(class_to_idx[class_name])

        if not paths:
            raise RuntimeError("No paths were generated from samples.")

        return paths, labels, class_to_idx
    

    def read_s3_bytes(self, path_bytes):
        path = path_bytes.decode("utf-8")
        assert path.startswith("s3://")
        no_scheme = path[len("s3://"):]
        bucket, key = no_scheme.split("/", 1)
        obj = self.s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()

    def load_and_preprocess(self, path: tf.Tensor, label: tf.Tensor):
        image_bytes = tf.io.read_file(path)
    
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])

        image = tf.image.resize(image, [self.image_size, self.image_size])
        image = tf.cast(image, tf.float32) / 255.0

        return image, tf.cast(label, tf.int32)

    def build_dataset(
        self,
        shuffle: bool = True,
        service: Optional[str] = None,
        job_name: str = "image-job",
    ) -> Tuple[tf.data.Dataset, Dict[str, int]]:
        samples = self.get_samples_from_s3()
        paths, labels, class_to_idx = self.flatten_samples(samples)

        self.logger.info("Found %d images across %d classes", len(paths), len(class_to_idx))

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

        ds = ds.map(self.load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size, drop_remainder=False)

        if service is not None:
            ds = ds.apply(
                tf.data.experimental.service.distribute(
                    processing_mode="distributed_epoch",
                    service=service,
                    job_name=job_name,
                )
            )

        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds, class_to_idx

