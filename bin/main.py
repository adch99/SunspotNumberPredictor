
# Getting the data
headers = ["Year",
           "Month",
           "Day",
           "Decimal Date",
           "Daily Total Sunspot Number",
           "Sunspot Number Stddev",
           "No of observations",
           "Definitive/Provisional"
]

data = pd.read_csv(filename, delimiter=";", names=headers)

# Hyper-Parameters
mean_length = 31
batch_size = 32
timesteps = 10
learning_rate = 0.001
epochs = 5
hidden_layer_size_1 = 64
hidden_layer_size_2 = 64
loss_func = "mse"

# Data Preprocessing
x, y = preprocess(data)
weights = gaussian(M=mean_length, std=mean_length, sym=True)
y = running_mean(y, weights)
x_slid, y_slid = sliding_window_main(x, y)
x_train, y_train, x_val, y_val, x_test, y_test = data_splitting_main(x_slid, y_slid)

net = create_network()


history = trainer(net, x_train, y_train, x_val, y_val, verbose=1)
var_train = np.var(y_train)
var_val = np.var(y_val)
print("Variance in y_train:", var_train)
print("Variance in y_val:", var_val)

plot_loss_vs_epoch(history)


# loss, val_loss = learning_curve()
