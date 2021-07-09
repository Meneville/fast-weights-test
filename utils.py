# ---------------------------------------------------------------------------
# 0. import
# ---------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import random
import os


# ---------------------------------------------------------------------------
# 1. functions
# ---------------------------------------------------------------------------
def reset_seed(SEED):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def loss_fn(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
