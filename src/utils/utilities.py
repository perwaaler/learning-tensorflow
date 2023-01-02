import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def mae(Y_test, Y_test_pred):
    """Mean average error between target and prediction."""
    Y_test = tf.squeeze(Y_test)
    Y_test_pred = tf.squeeze(Y_test_pred)
    return tf.metrics.mae(Y_test, Y_test_pred)


def mae_relative(Y_test, Y_test_pred):
    """Mean absolute percentage error between target and prediction."""
    Y_test = tf.squeeze(Y_test)
    Y_test_pred = tf.squeeze(Y_test_pred)
    mape = tf.metrics.mean_absolute_percentage_error(
        Y_test,
        Y_test_pred
    )/100
    return mape


def mse(Y_test, Y_test_pred):
    """Mean square error between target and prediction."""
    Y_test = tf.squeeze(Y_test)
    Y_test_pred = tf.squeeze(Y_test_pred)
    return tf.metrics.mse(Y_test, Y_test_pred)


def rmse(Y_test, Y_test_pred):
    """Root mean square error between target and prediction."""
    Y_test = tf.squeeze(Y_test)
    Y_test_pred = tf.squeeze(Y_test_pred)
    return tf.sqrt(tf.metrics.mse(Y_test, Y_test_pred))


def simulate_data(x_start=-100,
                  x_end=100,
                  frac_train=0.8,
                  power=1.0):
    """Simulates simple data that follows the rule y = (x + 10)**pow."""

    X = tf.range(x_start, x_end, 4, dtype="float32")
    X = tf.constant(X)
    Y = tf.math.pow(X*1 + 10, power)

    n = len(X)
    n_train = tf.cast(tf.round(n*frac_train), dtype="int32")
    X_train = X[:n_train]  # the first 40 elements are training data
    Y_train = Y[:n_train]
    X_test = X[n_train:]  # the last 10 elements are set aside as test data
    Y_test = Y[n_train:]

    return X_train, Y_train, X_test, Y_test


def plot_predictions(X_test,
                     Y_test,
                     Y_test_pred,
                     X_train=None,
                     Y_train=None,
                     Y_train_pred=None,
                     title=None):
    """Plots test predictions against true values."""

    if title == None:
        title = "predictions vs truth"

    if X_train != None:
        plt.scatter(X_train, Y_train, c="b", label="training truth")
        plt.scatter(X_train, Y_train_pred, c="g", label="training predictions")

    plt.scatter(X_test, Y_test, c="b", label="test truth")
    plt.scatter(X_test, Y_test_pred, c="purple", label="test predictions")

    plt.title(title)
    plt.legend()
    plt.show()


def get_auc_and_plot_roc(bin_target,
                         scores,
                         plot=False,
                         class_thr=None):
    """Computes the AUC and, optionally, plots ROC curve."""
    fpr, tpr, thresholds = roc_curve(bin_target, scores)
    auc = roc_auc_score(y_true=bin_target, y_score=scores)

    if plot == True:
        plt.plot(fpr, tpr, c="k")
        if class_thr == None:
            plt.title(f"The AUC is {auc:.2}")
        else:
            plt.title(f"ROC for prediction of charges >= {class_thr}.\n"
                      f"The AUC is {auc:.3}")
        plt.show()

    return auc, fpr, tpr, thresholds
