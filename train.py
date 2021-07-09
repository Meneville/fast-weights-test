# ---------------------------------------------------------------------------
# 0. import
# ---------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import _pickle as pickle
from dataset import DataGenerator
from model import fw_rnn_model
from utils import *

# ---------------------------------------------------------------------------
# 1. parameter
# ---------------------------------------------------------------------------
STEP_NUM = 9
ELEM_NUM = 26 + 10 + 1
BATCH_SZ = 128
HID_NUM = 50
SEED = 7777
MODEL = 'fw_rnn_model'

model_path = './checkpoint/' + MODEL
log_path = './log/' + MODEL
learning_rate = 1e-4
epochs = 1000

reset_seed(SEED)

# ---------------------------------------------------------------------------
# 2. Create Dataset
# ---------------------------------------------------------------------------
with open(os.path.join('data', 'train.p'), 'rb') as f:
    x_train, y_train = pickle.load(f)
with open(os.path.join('data', 'valid.p'), 'rb') as f:
    x_val, y_val = pickle.load(f)

train_gen = DataGenerator(x_train, y_train, BATCH_SZ, shuffle=False)
val_gen = DataGenerator(x_val, y_val, BATCH_SZ, shuffle=False)
# test_gen = DataGenerator(x_test, y_test, BATCH_SZ, shuffle=False)


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
opt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, clipnorm=0.1)
model = eval(MODEL)(BATCH_SZ, STEP_NUM, ELEM_NUM, HID_NUM)
model.summary()

model.compile(loss={'output': loss_fn},
              optimizer=opt,
              metrics=['accuracy'])
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy',
                                                verbose=1, save_best_only=True,
                                                mode='max', save_weights_only=True)
csv_logger = tf.keras.callbacks.CSVLogger(log_path, append=True, separator=',')
callbacks_list = [checkpoint, csv_logger]

model.fit(train_gen, epochs=epochs, verbose=1, callbacks=callbacks_list,
          validation_data=val_gen)
