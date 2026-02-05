import numpy as np  # Import the 'numpy' library as 'np'.
from sklearn.metrics import accuracy_score, mean_squared_error  # Import specific metrics from 'sklearn'.
from sklearn.utils import resample  # Import the 'resample' function from 'sklearn.utils'.

def bootstrap_accuracy(y, pred_y, n_iters=100):
    """
    Calculate bootstrap statistics for accuracy.

    :param y: True labels.
    :param pred_y: Predicted labels.
    :param n_iters: Number of bootstrap iterations (default is 100).
    :return: Mean and standard deviation of accuracy scores.
    """
    app = []
    for i in range(n_iters):
        sub_y, sub_pred_y = resample(y, pred_y)
        app.append(accuracy_score(sub_y, sub_pred_y))
    return (np.mean(app), np.std(app))

def bootstrap_mse(y, pred_y, n_iters=100):
    """
    Calculate bootstrap statistics for mean squared error (MSE).

    :param y: True values.
    :param pred_y: Predicted values.
    :param n_iters: Number of bootstrap iterations (default is 100).
    :return: Mean and standard deviation of MSE scores.
    """
    app = []
    for i in range(n_iters):
        sub_y, sub_pred_y = resample(y, pred_y)
        app.append(mean_squared_error(sub_y, sub_pred_y))
    return (np.mean(app), np.std(app))

# The following function is commented out, so it's not included in the comments.

'''
def bootstrap_ndcg(y, pred_y, n_iters=100):
    app = []
    for i in range(n_iters):
        sub_y, sub_pred_y = resample(y, pred_y)
        app.append(custom_ndcg(sub_y, sub_pred_y))
    return (np.mean(app), np.std(app))
'''
