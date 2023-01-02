# %% import dependencies
from get_data import DATA
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pylab as plt
import pandas as pd
import importlib
import numpy as np

import utils.utilities as mf
importlib.reload(mf)


# read insurance dataset

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


n_epochs = 150
history = model.fit(x_train,
                    y_train,
                    epochs=n_epochs,
                    verbose=False)

# predict on train and test set:
y_test_pred = model.predict(x_test)
y_train_pred = model.predict(x_train)

# create a binary prediction class defined by threshold exceedence:
class_thr = 30000
class_test = y_test >= class_thr
class_train = y_train >= class_thr
class_test_pred = y_test_pred >= class_thr
class_train_pred = y_train_pred >= class_thr

# get some test metrics:
mae_test = mf.mae(y_test_pred, y_test)
mae_train = mf.mae(y_train_pred, y_train)

print(f"the average test set error is {mae_test:.1f}\n"
      f"the average training set error is {mae_train:.1f}\n"
      f"after {n_epochs} epochs")

# %% plot results
auc = mf.get_auc_and_plot_roc(bin_target=class_test,
                              scores=y_test_pred,
                              plot=True,
                              class_thr=class_thr)

plt.scatter(y_test_pred, y_test)
plt.xlabel("target values")
plt.ylabel("predicted values")
plt.title("predicted vs target y-value (charges)")
plt.show()

training_progress = history.history
dict.keys(training_progress)
plt.plot(training_progress["mae"])
plt.title("mean-average-error during training")
plt.xlabel("epoch")
plt.ylabel("MAE")
plt.show()


# %% plot ROC and get test-AUC:


# alternatively:
# fpr, tpr, thresholds = roc_curve(class_test, y_test_pred)
# auc = roc_auc_score(y_true=class_test, y_score=y_test_pred)
# plt.plot(fpr, tpr)
# plt.title(f"ROC for prediction of charges >= {class_thr}.\n"
#           f"The AUC is {auc:.2}")
