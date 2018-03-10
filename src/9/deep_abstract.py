from beeprint import pp
import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report


wine_data = load_wine()

feat_data = wine_data['data']
labels = wine_data['target']

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(feat_data, labels, test_size=0.3, random_state=101)

from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)


def estimator():
    from tensorflow import estimator

    feat_cols = [tf.feature_column.numeric_column('x', shape=[13])]

    deep_model = estimator.DNNClassifier(
        hidden_units=[13, 13, 13],
        feature_columns=feat_cols,
        n_classes=3,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
    )

    train_input_fn = estimator.inputs.numpy_input_fn(
        x={"x": scaled_x_train},
        y=y_train,
        shuffle=True,
        batch_size=10,
        num_epochs=5
    )

    deep_model.train(input_fn=train_input_fn, steps=5000)

    test_input_fn = estimator.inputs.numpy_input_fn(
        x={"x": scaled_x_test},
        shuffle=False
    )

    preds = list(deep_model.predict(input_fn=test_input_fn))

    preds = [p["class_ids"][0] for p in preds]

    print(classification_report(y_test, preds))


def keras():
    from tensorflow.contrib.keras import models, layers, metrics, activations, losses, optimizers

    dnn_model = models.Sequential()

    dnn_model.add(layers.Dense(units=13, input_dim=13, activation="relu"))
    dnn_model.add(layers.Dense(units=13, activation="relu"))
    dnn_model.add(layers.Dense(units=13, activation="relu"))
    dnn_model.add(layers.Dense(units=3, activation="softmax"))

    dnn_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    dnn_model.fit(scaled_x_train, y_train, epochs=200)

    preds = dnn_model.predict_classes(scaled_x_test)

    print(classification_report(y_test, preds))


def layers():
    onehot_y_train = pd.get_dummies(y_train).values
    onehot_y_test = pd.get_dummies(y_test).values

    n_feat = 13
    n_hidden1 = 13
    n_hidden2 = 13
    n_out = 3
    learning_rate = 0.01
    steps = 1000

    from tensorflow.contrib.layers import fully_connected

    X = tf.placeholder(tf.float32, shape=[None, n_feat])
    y_true = tf.placeholder(tf.float32, shape=[None, 3])

    act_fn = tf.nn.relu

    hidden1 = fully_connected(X, n_hidden1, activation_fn=act_fn)
    hidden2 = fully_connected(hidden1, n_hidden2, activation_fn=act_fn)

    output = fully_connected(hidden2, n_out)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=output)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(steps):
            sess.run(train, feed_dict={X: scaled_x_train, y_true: onehot_y_train})

        logits = output.eval(feed_dict={X: scaled_x_test})

        preds = tf.argmax(logits, axis=1)

        results = preds.eval()

    print(classification_report(y_test, results))


layers()
