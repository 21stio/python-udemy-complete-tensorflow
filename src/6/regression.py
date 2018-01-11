import numpy as np
import pandas as pd
import plotly.offline as py

import tensorflow as tf
from beeprint import pp


x_data = np.linspace(0.0, 10.0, 1000000)

noise = np.random.randn(len(x_data))

y_true = (0.5 * x_data) + 5 + noise

df = pd.DataFrame(data={"y": y_true, "x": x_data})

#s = df.sample(n=250)
#py.plot({'data': [{'x': s["x"], 'y': s["y"], 'mode': 'markers'}], }, filename='/tmp/plot1.html')

batch_size = 8

m = tf.Variable(0.81)
b = tf.Variable(0.17)

xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

y_model = m*xph+b

error = tf.reduce_sum(tf.square(yph-y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batches = 1000

    for i in range(batches):
        rand_ind = np.random.randint(len(x_data), size=batch_size)

        feed = {xph:x_data[rand_ind], yph:y_true[rand_ind]}

        sess.run(train, feed_dict=feed)

    model_m, model_b = sess.run([m, b])

y_hat = x_data * model_m + model_b


s = df.sample(n=250)
py.plot({'data': [{'x': s["x"], 'y': s["y"], 'mode': 'markers'}, {'x': x_data, 'y': y_hat, 'mode': 'line'}], }, filename='/tmp/plot2.html')


