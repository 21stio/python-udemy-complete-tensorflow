import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd
from beeprint import pp
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("data/anonymized_data.csv")

x = df.values[:,:-1]
y = df.values[:,-1]

s = MinMaxScaler()
scaled_x = s.fit_transform(x)

n_in = scaled_x.shape[1]
n_hidden = 3
n_out = n_in
learning_rate = 0.01
n_steps = 1000

import tensorflow as tf

from tensorflow.contrib.layers import fully_connected

X = tf.placeholder(shape=[None, n_in], dtype=tf.float32)

hidden = fully_connected(X, n_hidden, activation_fn=None)
outputs = fully_connected(hidden, n_out, activation_fn=None)

loss = tf.reduce_mean(tf.square(outputs - X))

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(n_steps):
        sess.run(train, feed_dict={X: scaled_x})

    output_3d = hidden.eval(feed_dict={X: scaled_x})

x = output_3d[:, 0]
y = output_3d[:, 1]
z = output_3d[:, 2]

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