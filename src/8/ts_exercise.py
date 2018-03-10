import pandas as pd
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import tensorflow as tf
from beeprint import pp
from sklearn.preprocessing import MinMaxScaler

config = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    allow_soft_placement=True,
    device_count = {'CPU': 6}
)


df = pd.read_csv("./data/monthly-milk-production.csv", index_col="Month")

df.index = pd.to_datetime(df.index)

df.columns = ["y_true"]

train_df = df[:-12]
test_df = df[-12:]

# py.plot([
#     go.Scatter(x=df.index, y=df["y_true"], name=""),
#     go.Scatter(x=train_df.index, y=train_df["y_true"], name="train_df"),
#     go.Scatter(x=test_df.index, y=test_df["y_true"], name="test_s"),
# ], filename="/tmp/ts_exercise.html")

s = MinMaxScaler()
s.fit(train_df.values)

train_df = pd.DataFrame(s.transform(train_df.values), index=train_df.index, columns=train_df.columns)
test_df = pd.DataFrame(s.transform(test_df.values), index=test_df.index, columns=test_df.columns)


# py.plot([
#     go.Scatter(x=train_df.index, y=train_df["y_true"], name="train_df"),
#     go.Scatter(x=test_df.index, y=test_df["y_true"], name="test_s"),
# ], filename="/tmp/ts_exercise.html")

def next_batch(df, batch_size, steps, offset):
    rand_start = np.random.rand(batch_size, 1)

    data = df.values

    ts_start = rand_start * (len(data) - (steps + offset - 1))

    data = data[int(ts_start):]

    X_batch = data[:steps].reshape(-1, steps, 1)
    y_batch = data[offset:steps + offset].reshape(-1, steps, 1)

    return X_batch, y_batch


n_inputs = 1
n_outputs = 1
n_steps = 24
offset = 12
n_neurons_per_layer = 100
learning_rate = 0.03
n_iters = 4000
batch_size = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

# cell = tf.contrib.rnn.OutputProjectionWrapper(
#     tf.contrib.rnn.BasicRNNCell(num_units=n_neurons_per_layer, activation=tf.nn.relu),
#     output_size=n_outputs)

# n_layers = 3
# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons_per_layer)
#                                     for layer in range(n_layers)])

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons_per_layer, activation=tf.nn.elu),
    output_size=n_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()


def train_sess():
    with tf.Session(config=config) as sess:
        sess.run(init)

        for i in range(n_iters):
            X_batch, y_batch = next_batch(train_df, batch_size, n_steps, offset)

            sess.run(train, feed_dict={X: X_batch, y: y_batch})

            if i % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(i, "\tMSE:", mse)

        # Save Model for Later
        saver.save(sess, "./ts_exercise")


def pred_sess_true_seed():
    with tf.Session(config=config) as sess:
        saver.restore(sess, "./ts_exercise")

        pred = list(train_df[-n_steps:].values)

        pred_steps = int(12 / offset)
        for i in range(pred_steps):
            X_batch = np.array(pred[-n_steps:]).reshape(1, n_steps, 1)
            y_pred = sess.run(outputs, feed_dict={X: X_batch})

            pred = pred + list(y_pred[0, -offset:, 0])

        pred_df = pd.DataFrame(pred[n_steps:], index=test_df.index, columns=test_df.columns)
        seed_df = pd.DataFrame(pred[:n_steps], index=train_df[-n_steps:].index, columns=test_df.columns)

        py.plot([
            go.Scatter(x=train_df.index, y=train_df["y_true"], name="train_df"),
            go.Scatter(x=test_df.index, y=test_df["y_true"], name="test_s"),
            go.Scatter(x=pred_df.index, y=pred_df["y_true"], name="pred_s"),
            go.Scatter(x=seed_df.index, y=seed_df["y_true"], name="seed_s"),
        ], filename="/tmp/ts_exercise.html")


train_sess()
pred_sess_true_seed()
