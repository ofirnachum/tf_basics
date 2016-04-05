__doc__ = """Simple RNN."""

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class RNN(object):

    def __init__(self, num_emb, emb_dim, output_dim, hidden_dim, learning_rate=0.01):
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)

        self.embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
        self.recurrent_unit = self.create_recurrent_unit()
        self.h0 = tf.Variable(self.init_vector([self.hidden_dim]))
        self.W_out = tf.Variable(self.init_matrix([self.output_dim, self.hidden_dim]))
        self.b_out = tf.Variable(self.init_vector([self.output_dim]))

        self.x = tf.placeholder(tf.int32, shape=[None])  # sequence of indices
        self.y = tf.placeholder(tf.float32, shape=[self.output_dim])

        self.emb_x = tf.gather(self.embeddings, self.x)
        num_indices, = tf.unpack(tf.shape(self.x), 1)
        self.inputs = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=num_indices)
        self.inputs = self.inputs.unpack(self.emb_x)

        def _recurrence(t, h_tm1):
            inp = self.inputs.read(t)
            inp.set_shape([self.emb_dim, 1])
            h_t = self.recurrent_unit(
                    h_tm1,
                    tf.reshape(inp, [self.emb_dim, 1]))  # for some reason this reshape is necessary
            return (t + 1, h_t)

        time = tf.constant(0, dtype=tf.int32, name="time")
        state = self.h0
        _, final_hidden_state = control_flow_ops.While(
            cond=lambda t, _: t < num_indices,
            body=_recurrence,
            loop_vars=(time, state))

        self.pred_y = self.activation(
            tf.matmul(self.W_out, tf.reshape(final_hidden_state, [self.hidden_dim, 1]))
            + self.b_out)
        self.loss = self.loss_fn(self.y, self.pred_y)

        self.params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.grad = tf.gradients(self.loss, self.params)
        self.updates = opt.apply_gradients(zip(self.grad, self.params))

    def train_step(self, session, x, y):
        outputs = session.run([self.updates, self.loss],
                              feed_dict={self.x: x, self.y: y})
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def create_recurrent_unit(self):
        self.W_rec = tf.Variable(self.init_matrix([self.output_dim, self.emb_dim]))
        def unit(h_tm1, inp):
            return h_tm1 + tf.matmul(self.W_rec, inp)
        return unit

    def activation(self, inp):
        return tf.sigmoid(inp)

    def loss_fn(self, y, pred_y):
        return tf.reduce_sum(tf.square(y - pred_y))
