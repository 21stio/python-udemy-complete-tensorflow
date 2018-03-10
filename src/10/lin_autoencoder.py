import numpy as np
from beeprint import pp
import plotly.offline as py
import plotly.graph_objs as go

from sklearn.datasets import make_blobs


data = make_blobs(n_samples=100, n_features=3, centers=2, random_state=101)

pp(data)

from sklearn.preprocessing import MinMaxScaler


s = MinMaxScaler()
scaled_data = s.fit_transform(data[0])

x = scaled_data[:, 0]
y = scaled_data[:, 1]
z = scaled_data[:, 2]

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=z,
        colorscale='Viridis',
        opacity=0.8
    )
)])
py.plot(fig, filename='/tmp/3d-scatter-colorscale.html')

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


n_in = 3
n_hidden = 2
n_out = n_in

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None, n_in])

hidden = fully_connected(X, n_hidden, activation_fn=None)
outputs = fully_connected(hidden, n_out, activation_fn=None)

loss = tf.reduce_mean(tf.square(outputs - X))

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_steps = 1000

with tf.Session() as sess:
    sess.run(init)

    for i in range(n_steps):
        sess.run(train, feed_dict={X: scaled_data})

    output_2d = hidden.eval(feed_dict={X: scaled_data})


fig = go.Figure(data=[go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(
        size=12,
        color=z,
        colorscale='Viridis',
        opacity=0.8
    )
)])
py.plot(fig, filename='/tmp/2d-scatter-colorscale.html')
