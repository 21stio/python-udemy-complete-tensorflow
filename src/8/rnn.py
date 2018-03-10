import numpy as np
import tensorflow as tf
import plotly.offline as py
import plotly.graph_objs as go
from beeprint import pp


class TimeSeriesData():
    def __init__(self, num_points, xmin, xmax):

        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax - xmin) / num_points
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sin(x_series)

    def next_batch(self, batch_size, steps, return_batch_ts=False):

        # Grab a random starting point for each batch
        rand_start = np.random.rand(batch_size, 1)

        # Convert to be on time series
        ts_start = rand_start * (self.xmax - self.xmin - (steps * self.resolution))

        # Create batch Time Series on t axis
        batch_ts = ts_start + np.arange(0.0, steps + 1) * self.resolution

        # Create Y data for time series in the batches
        y_batch = np.sin(batch_ts)

        X_batch = y_batch[:, :-1].reshape(-1, steps, 1)
        #X_batch =  np.concatenate((X_batch,np.cos(-X_batch), np.cos(+X_batch), np.sin(-X_batch),np.arcsin(-X_batch)), axis=2)

        if return_batch_ts:
            return X_batch, y_batch[:, 1:].reshape(-1, steps, 1), batch_ts

        else:

            return X_batch, y_batch[:, 1:].reshape(-1, steps, 1)


ts_data = TimeSeriesData(500, 0, 20)

tf.reset_default_graph()

num_time_steps = 100

# Just one feature, the time series
num_inputs = 1
# 100 neuron layer, play with this
num_neurons = 100
# Just one output, predicted time series
num_outputs = 1
# learning rate, 0.0001 default, but you can play with this
learning_rate = 0.001
# how many iterations to go through (training steps), you can play with this
num_train_iterations = 2000
# Size of the batch of data
batch_size = 1

# ts_data.next_batch(batch_size, num_time_steps)

X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)

# cell = tf.contrib.rnn.OutputProjectionWrapper(
#     tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
#     output_size=num_outputs)


# n_neurons = 100
# n_layers = 3
# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
#           for layer in range(n_layers)])
#
# cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu)



# n_layers = 3
# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons)
#                                     for layer in range(n_layers)])

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y))  # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()


def train_sess():
    with tf.Session() as sess:
        sess.run(init)

        for iteration in range(num_train_iterations):

            X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)

            sess.run(train, feed_dict={X: X_batch, y: y_batch})

            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)

        # Save Model for Later
        saver.save(sess, "./rnn_time_series_model")


def pred_sess_zero_seed():
    with tf.Session() as sess:
        saver.restore(sess, "./rnn_time_series_model")

        zero_seq_seed = [0.0 for i in range(num_time_steps)]

        for iteration in range(len(ts_data.x_data) - num_time_steps):
            X_batch = np.array(zero_seq_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
            y_pred = sess.run(outputs, feed_dict={X: X_batch})

            zero_seq_seed.append(y_pred[0, -1, 0])

        py.plot([
            go.Scatter(x=ts_data.x_data, y=zero_seq_seed, name="pred"),
            go.Scatter(x=ts_data.x_data[:num_time_steps], y=zero_seq_seed[:num_time_steps], name="seed"),
            go.Scatter(x=ts_data.x_data, y=ts_data.y_true, name="true")
        ], filename='/tmp/rnn_zero_seed.html')


def pred_sess_true_seed():
    with tf.Session() as sess:
        saver.restore(sess, "./rnn_time_series_model")

        training_instance = list(ts_data.y_true[:num_time_steps])

        for iteration in range(len(ts_data.x_data) - num_time_steps):
            X_batch = np.array(training_instance[-num_time_steps:]).reshape(1, num_time_steps, 1)
            y_pred = sess.run(outputs, feed_dict={X: X_batch})

            training_instance.append(y_pred[0, -1, 0])

        py.plot([
            go.Scatter(x=ts_data.x_data, y=training_instance, name="pred"),
            go.Scatter(x=ts_data.x_data[:num_time_steps], y=training_instance[:num_time_steps], name="seed"),
            go.Scatter(x=ts_data.x_data, y=ts_data.y_true, name="true")
        ], filename='/tmp/rnn_true_seed.html')


train_sess()
pred_sess_true_seed()
