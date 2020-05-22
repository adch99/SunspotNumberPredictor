from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.hyperparams import *
import src.network as network


def plot_predictions(net, x_train, y_train, idx_train, x_val, y_val, idx_val):
    """
    Plot the prediction on the training and validation set.
    Inputs
    -------
    net: The neural network/model instance
    x_train: Training x data
    y_train: Training y data
    idx_train: Indexing set for the training data (usually the date)
    x_val: Validation x data
    y_val: Validation y data
    idx_val: Indexing set for the validation data (usually the date)

    Outputs
    -------
    Returns None
    Saves plot generated to
    img/[current date & time]_predicted_vs_actual_data.png
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 30))
    pred1 = net.predict(x_val, batch_size=batch_size)
    # print("pred1.shape:", pred1.shape)
    ax1.plot(idx_val, y_val, label="Actual Data", marker="+")
    ax1.plot(idx_val, pred1, label="Prediction", marker="o")
    # ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Sunspot Numbers")
    ax1.legend()
    ax1.set_title("Predicted vs Actual Validation Data")

    pred2 = net.predict(x_train, batch_size=batch_size)
    # print("pred2.shape:", pred2.shape)
    ax2.plot(idx_train, y_train, label="Actual Data", marker="+")
    ax2.plot(idx_train, pred2, label="Prediction", marker="o")
    # ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Sunspot Numbers")
    ax2.legend()
    ax2.set_title("Predicted vs Actual Training Data")

    plt.tight_layout()

    filename = "img/"
    filename += datetime.now().strftime("%y%m%d_%H%M")
    filename += "_predicted_vs_actual_data.png"
    fig.savefig(filename, format="png")

def plot_loss_vs_epoch(history, var_train, var_val, show=False):
    """
    Plot training & validation loss values.
    Inputs
    ------
    history: The network.history object after training the network
    var_train: Variance of the training y data
    var_val: Variance of the validation y data

    Outputs
    -------
    Returns None
    Saves plot generated to img/[current date & time]_model_loss.png
    """
    plt.figure(figsize=(10, 8))
    plt.grid(True)
    plt.plot(history.history['loss']/var_train, marker="o")
    plt.plot(history.history['val_loss']/var_val, marker="o")
    plt.title('Model Loss')
    plt.ylabel('Loss (Normalised to variance of dataset)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    # plt.ylim(bottom=0)
    filename = "img/"
    filename += datetime.now().strftime("%y%m%d_%H%M")
    filename += "_model_loss.png"
    plt.savefig(filename)

    if show:
        plt.show()


def learning_curve():
    """
    Generates a 'learning curve' i.e. the analysis for each data size 0 to full.
    Obsolete function.
    """
    loss = []
    val_loss = []
    data_size = []

    x_slid, y_slid = sliding_window_main(x, y)
    x_train, y_train, x_val, y_val, x_test, y_test = data_splitting_main(x_slid, y_slid)
    m_tot = x_train.shape[0]

    batch_step = 50
    try:
        for m in range(batch_size, m_tot, batch_step*batch_size):
            print("Training: ", m)
            net = create_network()
            history = trainer(net, x_train[:m], y_train[:m], x_val, y_val)
            loss.append(history.history["loss"][-1])
            val_loss.append(history.history["val_loss"][-1])
            data_size.append(m)

        print("Loss:", loss[-1])
        print()

    finally:
        plt.plot(data_size, loss, label="Loss", marker="o")
        plt.plot(data_size, val_loss, label="Val Loss", marker="o")
        plt.xlabel("m")
        plt.ylabel("Losses")
        plt.title("Model Loss")
        plt.legend()
        plt.savefig("img/" + datetime.now().strftime("%y%m%d_%H%M") + "_learning_curve.png")
        plt.show()
        plt.close()

    return loss, val_loss


def plot_weights(net, show=False):
    nrows = len(net.layers)
    cols = [len(layer.weights) for layer in net.layers] # One for colorbar
    ncols = max(cols)

    # print("cols:", cols)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

    cmap = mpl.cm.Blues

    # help(net.layers[0].weights[0].value())
    for i in range(nrows):
        for j in range(cols[i]):
            weights = net.layers[i].weights[j]
            name = weights.name
            wt_value = np.array(weights.value())
            if len(wt_value.shape) == 1:
                img = np.expand_dims(wt_value, axis=0)
            else:
                img = wt_value
            mappable = ax[i][j].imshow(img, cmap=cmap)
            ax[i][j].set_title(name)
            fig.colorbar(mappable, ax=ax[i][j])

    filename = "img/"
    filename += datetime.now().strftime("%y%m%d_%H%M")
    filename += "_weight_images.png"
    fig.savefig(filename)
    if show:
        plt.show()

def plot_recursive_predictions(predictor, x, y, idx, idx_end=None, show=False):

    if idx_end is None:
        idx_end = idx[-1]
    idx_step = np.average(np.diff(idx))
    x_start = x[:batch_size, :, :]
    idx_start = idx[:batch_size]

    args = (
        predictor,
        x_start,
        idx_start,
        idx_end,
        idx_step
    )

    npred = predictor.predict(x, batch_size=1)
    rpred, idx_rpred = network.predict_from_self(*args)

    fig = plt.figure(figsize=(10,8))
    plt.plot(idx, y, label="Actual Value", marker="+")
    plt.plot(idx_rpred, rpred, label="Recursive Prediction", marker="o")
    plt.plot(idx, npred, label="Normal Prediction", marker="x")

    plt.xlabel("Date")
    plt.ylabel("Normalised Sunspot Numbers")
    plt.title("predict_ahead = %d" % predict_ahead)
    plt.legend()

    filename = "img/"
    filename += datetime.now().strftime("%y%m%d_%H%M")
    filename += "_recursive_predictions.png"
    fig.savefig(filename)

    if show:
        plt.show()
