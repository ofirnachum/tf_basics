__doc__ = """Basic linear regression in TensorFlow."""

import tensorflow as tf


class Model(object):

    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)

        self.W = tf.Variable(tf.zeros([self.output_dim, self.input_dim]))
        self.b = tf.Variable(tf.zeros([self.output_dim]))

        self.x = tf.placeholder(tf.float32, shape=[self.input_dim])
        self.y = tf.placeholder(tf.float32, shape=[self.output_dim])

        self.pred_y = self.activation(
                tf.matmul(self.W, tf.reshape(self.x, [self.input_dim, 1])) + self.b)
        self.loss = self.loss_fn(self.y, self.pred_y)

        self.params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.grad = tf.gradients(self.loss, self.params)
        self.updates = opt.apply_gradients(zip(self.grad, self.params))

    def activation(self, inp):
        raise NotImplementedError

    def loss_fn(self, y, pred_y):
        raise NotImplementedError

    def train_step(self, session, x, y):
        outputs = session.run([self.updates, self.loss],
                              feed_dict={self.x: x, self.y: y})
        return outputs


class LinearRegression(Model):

    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        super(LinearRegression, self).__init__(
                input_dim, output_dim, learning_rate=learning_rate)

    def activation(self, inp):
        return inp

    def loss_fn(self, y, pred_y):
        return tf.reduce_sum(tf.square(y - pred_y))


class LogisticRegression(Model):

    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        super(LogisticRegression, self).__init__(
                input_dim, output_dim, learning_rate=learning_rate)

    def activation(self, inp):
        return tf.sigmoid(inp)

    def loss_fn(self, y, pred_y):
        return -tf.reduce_sum(y * tf.log(pred_y) + (1 - y) * tf.log(1 - pred_y))
