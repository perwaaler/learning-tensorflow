# %% import dependencies
from get_data import DATA
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pylab as plt
import pandas as pd
import importlib
import numpy as np

import utils.infrastructure as infra
import utils.utilities as mf
importlib.reload(mf)
importlib.reload(mf)


# %% plot data
plt.figure()
plt.scatter(np.arange(len(DATA["charges"])),
            DATA["charges"],
            marker=".")

# %% Convert categorical variables to one-hot-encoding
data = pd.get_dummies(DATA)

# %% Split into training and test set
y = data["charges"]
x = data.loc[:, data.columns != "charges"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42)

# %% build model
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.Input(shape=len(x_train.columns)),
    tf.keras.layers.Dense(20, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(30, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1)
])

history = model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.01),
    loss=tf.losses.mse,
    metrics="mae"
)

# define normalization/standardization transform for y-data:


def y_transform(x):
    # concave transformation:
    y = tf.math.sqrt(x)
    # vertical shift:
    global y_level_shift
    y_level_shift = tf.reduce_mean(y)
    y = y - y_level_shift
    return y


def y_transform_inv(y):
    x = y + y_level_shift
    x = tf.math.square(x)
    return x


data["charges"] = y_transform(DATA["charges"])

n_epochs = 150
history = model.fit(x_train,
                    y_transform(y_train),
                    epochs=n_epochs,
                    verbose=False)

# predict on train and test set:
y_test_pred = y_transform_inv(model.predict(x_test))
y_train_pred = y_transform_inv(model.predict(x_train))

# create a binary prediction class defined by threshold exceedence:
class_thr = 30000
class_test = y_test >= class_thr
class_train = y_train >= class_thr
class_test_pred = y_test_pred >= class_thr
class_train_pred = y_train_pred >= class_thr

# get some test metrics:
mae_test = mf.mae(y_test_pred, y_test)
mae_train = mf.mae(y_train_pred, y_train)
mae_relative_test = mf.mae_relative(y_train_pred, y_train)

print(f"the average test set error is {mae_test:.1f}\n"
      f"the average training set error is {mae_train:.1f}\n"
      f"after {n_epochs} epochs")

# %% plot results
auc = mf.get_auc_and_plot_roc(bin_target=class_test,
                              scores=y_test_pred,
                              plot=True,
                              class_thr=class_thr)[0]

plt.figure()
plt.scatter(y_test_pred, y_test)
plt.xlabel("target values")
plt.ylabel("predicted values")
plt.title(f"predicted vs target Y = insurance cost\n"
          f"MAE: {mae_test:.2f}. MAE relative: {mae_relative_test:.2}")

training_progress = history.history
dict.keys(training_progress)
plt.figure()
plt.plot(training_progress["mae"])
plt.title("mean-average-error during training")
plt.xlabel("epoch")
plt.ylabel("MAE")

# %% save metrics in yaml file:

metrics = {"auc": float(auc),
           "mae": float(mae_test.numpy()),
           "mae_relative": float(mae_relative_test.numpy())}

infra.save_to_yaml(
    "/home/per/medsensio/learning/tensorflow/metrics/metrics.yml", metrics)

plt.show()
