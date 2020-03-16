# Data Preprocessing
import numpy as np
from scipy.interpolate import interp1d
from src.hyperparams import *

def preprocess(data):
    # Set unknown numbers to NaN first
    mask = data["Daily Total Sunspot Number"] == -1
    data.loc[mask, "Daily Total Sunspot Number"] = np.nan

    # Remove the starting NaNs. Can't do anything about them.
    i = 0
    while i < data.shape[0]:
      if not mask[i]:
        break
      i += 1

    data = data.drop(range(i))
    mask = mask[i:]

    # Data Variables
    x = np.array(data["Decimal Date"])
    y = np.array(data["Daily Total Sunspot Number"])

    # Interpolate and fill in the missing data
    interp = interp1d(x[~mask], y[~mask], kind="linear", bounds_error=False)
    data.loc[mask, "Daily Total Sunspot Number"] = interp(x[mask])

    # Update the variables
    x = np.array(data["Decimal Date"])
    y = np.array(data["Daily Total Sunspot Number"])

    # Normalise the data
    y = (y - np.mean(y)) / (6*np.sqrt(np.var(y)))
    x = x - x.min()

    print("x.shape:", x.shape)
    print("y.shape:", y.shape)

    return x, y

# Smoothen the data by taking a running mean.
def running_mean(y, weights):
    return np.convolve(y, weights, mode="valid")

def sliding_window(x, timesteps):
    shape = (x.shape[0]-timesteps, timesteps, n)
    x_slid = np.zeros(shape)

    for i in range(shape[0]):
        x_slid[i] = x[i: i+timesteps].reshape(timesteps, n)

    return x_slid

def to_sliding_window(x, y, timesteps):
    # Gets the sliding window version of x and cuts the y data appropriately
    xnew = sliding_window(x, timesteps)
    #print(xnew)
    ynew = y[timesteps:]
    return xnew, ynew

def batch_adjustment(x, y, batch_size):
    assert x.shape[0] == y.shape[0], "x and y are not of same length (dim 0) x:%s y:%s" % (x.shape, y.shape)
    l = x.shape[0]
    lnew = int((l // batch_size) * batch_size)
    return x[:lnew], y[:lnew]

def sliding_window_main(x, y):
    """
    Just a wrapper function to allow use during iteration while making
    learning curves.
    """
    x_slid, y_slid = to_sliding_window(x, y, timesteps)
    reshape_2 = lambda y: np.reshape(y, (y.shape[0], n))
    y_slid = reshape_2(y_slid)

    print("x_slid.shape:", x_slid.shape)
    print("y_slid.shape:", y_slid.shape)

    return x_slid, y_slid

# Data Splitting
def split_data(x, y, ratio):
    """
    Splits the data into training, validation and testing sets
    for x and y in the given ratio.
    """
    m = x.shape[0]
    splitter = np.cumsum(ratio)
    train_start = 0
    val_start = batch_size * ((splitter[0] * m) // batch_size)
    test_start = batch_size * ((splitter[1] * m) // batch_size)
    test_end = batch_size * ((splitter[2] * m) // batch_size)

    val_start = int(val_start)
    test_start = int(test_start)
    test_end = int(test_end)

    split = ( x[train_start:val_start, :], y[train_start:val_start, :],
           x[val_start:test_start, :], y[val_start:test_start, :],
           x[test_start:test_end, :], y[test_start:test_end, :]
    )

    return split

def data_splitting_main(x_slid, y_slid):
    """
    Just a wrapper function to allow use during iteration while making
    learning curves.
    """
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_slid, y_slid, data_split_ratio)

    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('x_val.shape: ', x_val.shape)
    print('y_val.shape: ', y_val.shape)
    print('x_test.shape: ', x_test.shape)
    print('y_test.shape: ', y_test.shape)
    print()

    return x_train, y_train, x_val, y_val, x_test, y_test


