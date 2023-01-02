# %% Import dependencies
import yaml
import os
import utils.utilities as mf
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import utils.utilities as utilities

from get_paths import DATA_DIR, PROJECT_DIR, PARAMETERS_DIR

path_saved_model = "/home/per/medsensio/learning/tensorflow/models"

# use the reload function to reload the module so that new functions are found:
importlib.reload(utilities)

print(tf.__version__)

# %% fit a simple "deep learning" regression model to simulated data =============
# %% create data
X = np.array([-7, -4, -1, 2, 5, 8, 11, 14.])
Y = np.array([3, 6, 9, 12, 15, 18, 21, 24.])
print(X.shape)
print(Y.shape)

plt.scatter(X, Y)

# check that suspected formula is indeed the case:
print(Y == X + 10)

# %% convert arrays to tensors
# convert np arrays to tensors:
X = tf.constant(X)
Y = tf.constant(Y)

# %% 1. Create the model
model = tf.keras.Sequential([tf.keras.Input(shape=(1)),
                             tf.keras.layers.Dense(1)])

# %% 2. Compile (bring together) the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              metrics=["mae"])

# %% 3. Fit the model
tf.random.set_seed(1)
model.fit(X, Y, epochs=100)

# %% Make prediction using trained model
Y_pred = np.round(model.predict(X), 2)
print(f"the model predictions are\n{tf.transpose(Y_pred)}"
      f"the actual values are\n{Y}")

plt.scatter(Y_pred, Y)

# %% predict on a new x-value
print(f"the test prediction is {model.predict([17.])}, and the actual"
      f"y-value is {27.0}")


# %% The 3 sets; training, validation, and test set ==============================
# %% simulate larger dataset
X = tf.range(-100, 100, 4, dtype=tf.float32)
X = tf.constant(X)
Y = X*1 + 10
plt.scatter(X, Y)

X_train = X[:40]  # the first 40 elements are training data
Y_train = Y[:40]
X_test = X[40:]  # the last 10 elements are set aside as test data
Y_test = Y[40:]

plt.figure(figsize=(10, 7))
plt.scatter(X_train, Y_train, c="b", label="Training data")
plt.scatter(X_test, Y_test, c="g", label="Testing data")
plt.legend()

# %% create a model and plot predictions against ground truth

# create model arcitechture:
model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=1, name="input"),
                             tf.keras.layers.Dense(100, name="output_layer"),
                             tf.keras.layers.Dense(1, name="regression_layer")],
                            name="my_first_model")

# compile model:
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
              loss=tf.keras.losses.mse,
              metrics=tf.keras.metrics.mse)

model.summary()

# fit the model:
model.fit(X_train, Y_train, epochs=50, verbose=0, batch_size=None)

# test set predictions:
Y_test_pred = model.predict(X_test)
Y_train_pred = model.predict(X_train)
# Need to make dimension match, since one is a vector and the other is a matrix  with one column:
# use reshape:
# Y_train_pred = tf.reshape(Y_train_pred, shape=Y_train_pred.size)
# Y_test_pred = tf.reshape(Y_test_pred, shape=Y_test_pred.size)
# easy way:
# you can also use Y_train_pred.squeeze()
Y_train_pred = tf.squeeze(Y_train_pred)
Y_test_pred = tf.squeeze(Y_test_pred)
e_test = tf.transpose(Y_test_pred) - Y_test
mae_test = tf.keras.metrics.mean_absolute_error(Y_test,
                                                Y_test_pred)

print(f"the mean-absolute-error on the test set is {mae_test}")

# plot model
tf.keras.utils.plot_model(model=model,
                          show_shapes=True)


mf.plot_predictions(X_test,
                    Y_test,
                    Y_test_pred,
                    X_train,
                    Y_train,
                    Y_train_pred)


# %% Evaluation metrics ==========================================================
# test set predictions:
mae_test = tf.keras.metrics.mean_absolute_error(Y_test,
                                                Y_test_pred)
mse_test = tf.keras.metrics.mse(Y_test,
                                Y_test_pred)
mse_test1 = tf.metrics.mse(Y_test, Y_test_pred)

rmse_test = tf.sqrt(mse_test)
# NOTE: to compute metrics, you can use either the tf.metrics or the tf.keras.or the tf.keras.losses.metrics library.


print(f"mean absolute error is {mae_test:.3}\n"
      f"mean squared error is {mse_test:.3} and {mse_test1:.3}\n"
      f"the root-mean-squared error is {rmse_test:.3}")

# use custom function defined utilities.py
rmse_test = mf.rmse(Y_test, Y_test_pred)
mae_test = mf.mae(Y_test, Y_test_pred)
# NOTE: after defining a new function in utilities.py you have to reload the module (see import section of this script).


# %% Experiment: try out many different models ========================================

# simulate data:
X_train, Y_train, X_test, Y_test = mf.simulate_data(power=2)

model0 = tf.keras.Sequential([tf.keras.Input(shape=1),
                              tf.keras.layers.Dense(1)],
                             name="simple_model")
# create model arcitechture:
model1 = tf.keras.Sequential([
    tf.keras.Input(shape=1, name="input_layer"),
    tf.keras.layers.Dense(200,
                          name="hidden_layer_1",
                          activation=tf.keras.activations.gelu),
    tf.keras.layers.Dense(100,
                          name="hidden_layer_2",
                          activation=tf.keras.activations.linear),
    tf.keras.layers.Dense(1, name="regression_layer")],
    name="2_layer_model")

model0.summary()
model1.summary()

model0.compile(loss=tf.metrics.mse,
               metrics=tf.metrics.mse,
               optimizer=tf.optimizers.Adam(learning_rate=0.02))
model1.compile(loss=tf.metrics.mse,
               metrics=tf.metrics.mse,
               optimizer=tf.optimizers.Adam(learning_rate=0.02))

# fit models and get predictions:
model0.fit(X_train, Y_train, epochs=100)
Y_train_pred_ex0 = model0.predict(X_train)
Y_test_pred_ex0 = model0.predict(X_test)
rmse_ex0 = mf.rmse(Y_test_pred_ex0, Y_test)

model1.fit(X_train, Y_train, epochs=100)
Y_train_pred_ex1 = model1.predict(X_train)
Y_test_pred_ex1 = model1.predict(X_test)
rmse_ex1 = mf.rmse(Y_test_pred_ex1, Y_test)

model1.fit(X_train, Y_train, epochs=500)
Y_train_pred_ex2 = model1.predict(X_train)
Y_test_pred_ex2 = model1.predict(X_test)
rmse_ex2 = mf.rmse(Y_test_pred_ex2, Y_test)

print(f"the rmse for experiment 0 is: {rmse_ex0:.2}"
      f"the rmse for experiment 1 is: {rmse_ex1:.2}"
      f"the rmse for experiment 2 is: {rmse_ex2:.2}")

# plot results:
mf.plot_predictions(X_test,
                    Y_test,
                    Y_test_pred_ex0,
                    X_train,
                    Y_train,
                    Y_train_pred_ex0,
                    title=f"1 layer ANN. RMSE-test = {rmse_ex0:.2}")

mf.plot_predictions(X_test,
                    Y_test,
                    Y_test_pred_ex1,
                    X_train,
                    Y_train,
                    Y_train_pred_ex1,
                    title=f"2 layers, 100 epochs. RMSE-test = {rmse_ex1:.2}")

mf.plot_predictions(X_test,
                    Y_test,
                    Y_test_pred_ex2,
                    X_train,
                    Y_train,
                    Y_train_pred_ex2,
                    title=f"2 layers, 500 epochs. RMSE-test = {rmse_ex2:.2}")


# %% compare results in a pandas dataframe
model_results = [["model0", rmse_ex0.numpy()],
                 ["model1", rmse_ex1.numpy()],
                 ["model2", rmse_ex2.numpy()]]

all_results = pd.DataFrame(model_results, columns=["model", "RMSE"])
all_results

# %% save model
model1.save(os.path.join(path_saved_model, "ANN_1Layer.py"))
model1.save(os.path.join(path_saved_model, "ANN_2Layer_ActGelu.py"))

# %% save as HDF5
model1.save(os.path.join(path_saved_model, "ANN_1Layer.h5"))

# %% Loading in a saved model
# load a model stored as a .py format:
model1 = tf.keras.models.load_model(os.path.join(
    path_saved_model, "ANN_2Layer_ActGelu.py")
)
# load a model stored in the .h5 format:
model2 = tf.keras.models.load_model(os.path.join(
    path_saved_model, "ANN_1Layer.h5")
)
# check model summary:
model1.summary()


# %% Load data
