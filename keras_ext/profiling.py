import tensorflow as tf
from tensorflow.python.client import timeline


def get_profiling_execute_kwargs(enabled=True):
    execute_kwargs = dict(
        options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        run_metadata=tf.RunMetadata()
    )
    return execute_kwargs

def save_timeline(fname, run_metadata):
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open(fname, 'w') as f:
        f.write(ctf)
