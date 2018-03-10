import pandas as pd
import numpy as np

from beeprint import pp
from numpy.ma import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("data/cal_housing_clean.csv")

y_df = df["medianHouseValue"]
x_df = df.drop("medianHouseValue", axis=1)

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.30)

s = MinMaxScaler(copy=True, feature_range=(0, 1))

s.fit(X_train)

X_train_df = pd.DataFrame(s.transform(X_train), columns=x_df.columns, index=X_train.index)
X_test_df = pd.DataFrame(s.transform(X_test), columns=x_df.columns, index=X_test.index)

import tensorflow as tf


f1 = tf.feature_column.numeric_column('housingMedianAge')
f2 = tf.feature_column.numeric_column('totalRooms')
f3 = tf.feature_column.numeric_column('totalBedrooms')
f4 = tf.feature_column.numeric_column('population')
f5 = tf.feature_column.numeric_column('households')
f6 = tf.feature_column.numeric_column('medianIncome')

fs = [f1, f2, f3, f4, f5, f6]

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train_df, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

m = tf.estimator.DNNRegressor(hidden_units=[6,6,6], feature_columns=fs)

m.train(input_fn=input_func, steps=20000)

# eval_input_func = tf.estimator.inputs.pandas_input_fn(
#     x=X_test_df,
#     y=y_test,
#     batch_size=10,
#     num_epochs=1,
#     shuffle=False
# )
#
# r = m.evaluate(eval_input_func, steps=100)

# pp(r)

pred_input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_test_df,
    shuffle=False,
    batch_size=10,
    num_epochs=1
)

y_hat = [y["predictions"][0] for y in m.predict(input_fn=pred_input_func)]

rmse = np.mean((y_test.values - y_hat) ** 2) ** 0.5

mse = mean_squared_error(y_test.values, y_hat)

pp(mse)
pp(rmse)
