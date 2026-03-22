import json
import os
import socket
import statistics
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import tensorflow as tf

from s3_image_dataset import S3ImageDatasetBuilder


@dataclass
class TrainerMetrics:
    trainer_name: str
    trainer_id: str
    mode: str
    model_name: str
    service: Optional[str]
    steps: int
    batch_size: int
    num_examples: int
    total_time_sec: float
    first_batch_time_sec: float
    avg_fetch_time_sec: float
    p50_fetch_time_sec: float
    p95_fetch_time_sec: float
    avg_step_time_sec: float
    p50_step_time_sec: float
    p95_step_time_sec: float
    avg_compute_time_sec: float
    examples_per_sec: float
    final_loss: float
    hostname: str
    pid: int
    started_at_unix: float

    def to_dict(self):
        return asdict(self)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = int(q * (len(values) - 1))
    return sorted(values)[idx]


def make_model(name: str, num_classes: int, image_size: int) -> tf.keras.Model:
    if name == "mobilenet_v2":
        return tf.keras.applications.MobileNetV2(
            weights=None,
            classes=num_classes,
            input_shape=(image_size, image_size, 3),
        )
    if name == "resnet50":
        return tf.keras.applications.ResNet50(
            weights=None,
            classes=num_classes,
            input_shape=(image_size, image_size, 3),
        )
    raise ValueError(f"Unknown model: {name}")


def make_dataset(
    *,
    path: str,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    service: Optional[str],
    job_name: str,
    repeat: bool,
    trainer_id: Optional[str],
    use_cross_trainer_cache: bool,
):
    builder = S3ImageDatasetBuilder(
        path=path,
        image_size=image_size,
        batch_size=batch_size,
    )

    ds, class_to_idx = builder.build_dataset(
        shuffle=shuffle,
        service=service,
        job_name=job_name,
        repeat=repeat,
        trainer_id=trainer_id,
        use_cross_trainer_cache=use_cross_trainer_cache,
        reshuffle_each_iteration=False,
    )
    return ds, class_to_idx


@tf.function
def train_step(model, optimizer, loss_fn, images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = loss_fn(labels, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def run_trainer_benchmark(
    *,
    trainer_name: str,
    trainer_id: str,
    mode: str,
    model_name: str,
    path: str,
    image_size: int,
    batch_size: int,
    steps: int,
    shuffle: bool,
    service: Optional[str],
    job_name: str,
    repeat: bool,
    use_cross_trainer_cache: bool,
    learning_rate: float = 1e-3,
) -> TrainerMetrics:
    ds, class_to_idx = make_dataset(
        path=path,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        service=service,
        job_name=job_name,
        repeat=repeat,
        trainer_id=trainer_id,
        use_cross_trainer_cache=use_cross_trainer_cache,
    )

    model = make_model(model_name, len(class_to_idx), image_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    it = iter(ds)
    started_at = time.time()
    t0 = time.perf_counter()

    fetch_times = []
    step_times = []
    compute_times = []
    first_batch_time = None
    num_examples = 0
    final_loss = 0.0

    for step in range(steps):
        s0 = time.perf_counter()
        images, labels = next(it)
        s_fetch = time.perf_counter()

        if step == 0:
            first_batch_time = s_fetch - t0
            print(f"{trainer_name}: first batch images={images.shape} labels={labels.shape}")

        loss = train_step(model, optimizer, loss_fn, images, labels)
        s1 = time.perf_counter()

        fetch_time = s_fetch - s0
        step_time = s1 - s0
        compute_time = s1 - s_fetch

        fetch_times.append(fetch_time)
        step_times.append(step_time)
        compute_times.append(compute_time)

        batch_n = int(images.shape[0])
        num_examples += batch_n
        final_loss = float(loss.numpy())

        print(
            f"{trainer_name}: step={step:04d} "
            f"fetch={fetch_time:.4f}s "
            f"compute={compute_time:.4f}s "
            f"step={step_time:.4f}s "
            f"loss={final_loss:.4f}"
        )

    total_time = time.perf_counter() - t0

    return TrainerMetrics(
        trainer_name=trainer_name,
        trainer_id=trainer_id,
        mode=mode,
        model_name=model_name,
        service=service,
        steps=steps,
        batch_size=batch_size,
        num_examples=num_examples,
        total_time_sec=total_time,
        first_batch_time_sec=first_batch_time or 0.0,
        avg_fetch_time_sec=statistics.mean(fetch_times) if fetch_times else 0.0,
        p50_fetch_time_sec=percentile(fetch_times, 0.50),
        p95_fetch_time_sec=percentile(fetch_times, 0.95),
        avg_step_time_sec=statistics.mean(step_times) if step_times else 0.0,
        p50_step_time_sec=percentile(step_times, 0.50),
        p95_step_time_sec=percentile(step_times, 0.95),
        avg_compute_time_sec=statistics.mean(compute_times) if compute_times else 0.0,
        examples_per_sec=(num_examples / total_time) if total_time > 0 else 0.0,
        final_loss=final_loss,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        started_at_unix=started_at,
    )


def write_metrics_json(metrics: TrainerMetrics, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2, sort_keys=True)
    print(f"Wrote metrics to {output_path}")