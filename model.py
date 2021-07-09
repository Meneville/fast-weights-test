# ---------------------------------------------------------------------------
# 0. import
# ---------------------------------------------------------------------------
import tensorflow as tf
import numpy as np


# ---------------------------------------------------------------------------
# 1. FW model for associative retrieval
# ---------------------------------------------------------------------------
class fw_rnn_cell(tf.keras.layers.Layer):
    def __init__(self,
                 num_class,
                 hidden_units,
                 fw_S=1,
                 fw_l=0.95,
                 fw_e=0.5,
                 **kwargs):
        self.num_class = num_class
        self.units = hidden_units
        self.S = fw_S
        self.l = fw_l
        self.e = fw_e

        self.W_x = tf.Variable(tf.random.uniform(
            [self.num_class, self.units],
            -np.sqrt(2.0 / self.num_class),
            np.sqrt(2.0 / self.num_class)),
            dtype=tf.float32)

        self.b_x = tf.Variable(tf.zeros(
            [self.units]),
            dtype=tf.float32)

        self.W_h = tf.Variable(
            initial_value=0.05 * np.identity(self.units),
            dtype=tf.float32)

        # softmax weights (proper initialization)
        self.W_softmax = tf.Variable(tf.random.uniform(
            [self.units, self.num_class],
            -np.sqrt(2.0 / self.units),
            np.sqrt(2.0 / self.units)),
            dtype=tf.float32)
        self.b_softmax = tf.Variable(tf.zeros(
            [self.num_class]),
            dtype=tf.float32)

        # scale and shift for layer norm
        self.gain = tf.Variable(tf.ones(
            [self.units]),
            dtype=tf.float32)
        self.bias = tf.Variable(tf.zeros(
            [self.units]),
            dtype=tf.float32)

        super(fw_rnn_cell, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        input_shape = inputs.shape
        batch_size = input_shape[0]
        num_step = input_shape[1]

        self.h = tf.zeros([batch_size, self.units], dtype=tf.float32)
        self.A = tf.zeros([batch_size, self.units, self.units], dtype=tf.float32)

        for t in range(0, num_step):
            # hidden state (preliminary vector)
            self.h = tf.nn.relu((tf.matmul(inputs[:, t, :], self.W_x) + self.b_x) +
                                (tf.matmul(self.h, self.W_h)))

            self.h_s = tf.reshape(self.h, [batch_size, 1, self.units])
            self.A = tf.add(tf.scalar_mul(self.l, self.A),
                            tf.scalar_mul(self.e, tf.matmul(tf.transpose(
                                self.h_s, [0, 2, 1]), self.h_s)))
            # Loop for S steps
            for _ in range(self.S):
                self.h_s = tf.reshape(
                    tf.matmul(inputs[:, t, :], self.W_x) + self.b_x,
                    tf.shape(self.h_s)) + tf.reshape(
                    tf.matmul(self.h, self.W_h), tf.shape(self.h_s)) + \
                           tf.matmul(self.h_s, self.A)

                # Apply layer norm
                mu = tf.reduce_mean(self.h_s, axis=2)  # each sample
                sigma = tf.sqrt(tf.reduce_mean(tf.square(tf.squeeze(self.h_s) - mu), axis=1))

                self.h_s = tf.divide(tf.multiply(self.gain, (tf.squeeze(self.h_s) - mu)), tf.expand_dims(sigma, -1)) + \
                           self.bias

                # Apply nonlinearity
                self.h_s = tf.nn.relu(self.h_s)

            # Reshape h_s into h
            self.h = tf.reshape(self.h_s, [batch_size, self.units])

        output = tf.matmul(self.h, self.W_softmax) + self.b_softmax

        return output


def fw_rnn_model(batch_sz, step_num, elem_num, hidden_dim):
    inputs = tf.keras.layers.Input(batch_shape=(batch_sz, step_num, elem_num))
    output = fw_rnn_cell(elem_num, hidden_dim, name='output')(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model


if __name__ == '__main__':
    model = fw_rnn_model(128, 9, 37, 50)
    model.summary()
    input = tf.zeros([128, 9, 37])
    out = model(input)
    y = tf.zeros([128, 37])
    print(out.shape)
