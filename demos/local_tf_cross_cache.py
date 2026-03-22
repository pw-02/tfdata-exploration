import threading
import time
import tensorflow as tf

# -----------------------------
# Start local tf.data service
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

SERVICE = "grpc://localhost:5000"

# -----------------------------
# Expensive dataset
# Must be infinite for this feature
# -----------------------------
def slow_map(x):
    def _work(v):
        time.sleep(0.02)  # simulate expensive preprocessing
        return v
    y = tf.py_function(_work, [x], Tout=tf.int64)
    y.set_shape(())
    return y

def make_dataset(trainer_id: str):
    ds = tf.data.Dataset.range(1000).repeat()
    ds = ds.map(slow_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(32)

    ds = ds.apply(
        tf.data.experimental.service.distribute(
            processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
            service=SERVICE,
            job_name="shared_job",
            cross_trainer_cache=tf.data.experimental.service.CrossTrainerCache(
                trainer_id=trainer_id
            ),
        )
    )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def run_trainer(name, trainer_id, steps=20):
    ds = make_dataset(trainer_id)
    start = time.time()
    for i, batch in enumerate(ds.take(steps)):
        if i == 0:
            print(f"{name}: first batch shape={batch.shape}")
    elapsed = time.time() - start
    print(f"{name}: {steps} steps in {elapsed:.2f}s")

t1 = threading.Thread(target=run_trainer, args=("trainerA", "A"))
t2 = threading.Thread(target=run_trainer, args=("trainerB", "B"))

t1.start()
time.sleep(1.0)  # stagger start so trainerA fills cache first
t2.start()

t1.join()
t2.join()