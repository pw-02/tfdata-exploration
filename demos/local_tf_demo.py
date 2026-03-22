import tensorflow as tf
import time

# -----------------------------
# 1) Create a toy dataset
# -----------------------------
def slow_map(x):
    # simulate expensive preprocessing
    def _work(v):
        time.sleep(0.01)
        return v * v
    y = tf.py_function(_work, [x], Tout=tf.int64)
    y.set_shape(())
    return y

ds = tf.data.Dataset.range(1000)
ds = ds.map(slow_map, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(32)
ds = ds.prefetch(tf.data.AUTOTUNE)

# -----------------------------
# 2) Dummy "training loop"
# -----------------------------
start = time.time()

for i, batch in enumerate(ds):
    if i % 10 == 0:
        print(f"Step {i}, first element: {batch[0].numpy()}")

end = time.time()

print(f"\nTotal time: {end - start:.2f} seconds")