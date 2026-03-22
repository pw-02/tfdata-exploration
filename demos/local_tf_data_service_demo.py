import time
import tensorflow as tf

# -----------------------------
# 1) Start a local tf.data service
# -----------------------------
dispatcher = tf.data.experimental.service.DispatchServer(
    tf.data.experimental.service.DispatcherConfig(
        port=5000,
        work_dir="/tmp/tf_data_dispatcher"
    )
)

worker1 = tf.data.experimental.service.WorkerServer(
    tf.data.experimental.service.WorkerConfig(
        dispatcher_address="localhost:5000",
        port=5001
    )
)

worker2 = tf.data.experimental.service.WorkerServer(
    tf.data.experimental.service.WorkerConfig(
        dispatcher_address="localhost:5000",
        port=5002
    )
)

service = "grpc://localhost:5000"

print("Dispatcher:", dispatcher.target)
print("tf.data service endpoint:", service)

# -----------------------------
# 2) Build a toy dataset
#    Everything before distribute(...)
#    can run on tf.data service workers
# -----------------------------
def slow_map(x):
    # tf.py_function is only for demo purposes here.
    # It makes the step visibly "expensive".
    def _sleep_and_square(v):
        time.sleep(0.01)
        return v * v
    y = tf.py_function(_sleep_and_square, [x], Tout=tf.int64)
    y.set_shape(())
    return y

ds = tf.data.Dataset.range(1000)
ds = ds.map(slow_map, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(32)

# Send the upstream pipeline to tf.data service
ds = ds.apply(
    tf.data.experimental.service.distribute(
        processing_mode="distributed_epoch",
        service=service,
        job_name="demo_job"
    )
)

# Keep normal downstream consumer-side ops local
ds = ds.prefetch(tf.data.AUTOTUNE)

# -----------------------------
# 3) Consume it like a normal dataset
# -----------------------------
for i, batch in enumerate(ds.take(10)):
    print(f"Batch {i}: shape={batch.shape}, first={batch[0].numpy()}")

print("Done.")