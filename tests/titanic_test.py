import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tensorflow as tf
from keras import Sequential, layers, optimizers
from keras.losses import BinaryCrossentropy

tf.keras.utils.disable_interactive_logging()

import titanic_data
import numpy as np
from numpyimpl.NN import NN, Layer
import matplotlib.pyplot as plt


from numpyimpl.optimizers import Adam


np_raw = titanic_data.load_train_data()

X_train = np_raw[:600]
Y_train = X_train[:, 0]

X_train = X_train[:, 1:]

X_test = np_raw[600:]
Y_test = X_test[:, 0]
X_test = X_test[:, 1:]

def run_experiment(n):
    # np.argmax(Y_pred, axis=1)
    exp_results = []
    for i in range(n):
        start = time.time()
        net = NN(layers=[
            Layer(7, 15, activation='relu'),
            Layer(15, 1, activation='sigmoid'),
        ], loss='binary_crossentropy', optimizer=Adam())

        net.fit(X_train, Y_train, epochs=1000, learning_rate=0.01, sample_size=100, include_logs=False)

        Y_pred = net.predict(X_test)
        time_n = (time.time()) - start

        wrong = 101 - Y_pred[Y_pred > 0.5].shape[0]

        exp_results.append(np.array([time_n, wrong]))
    return exp_results

def run_experiment_tf(n):
    # np.argmax(Y_pred, axis=1)
    exp_results = []
    for i in range(n):
        # Create the model using the Sequential API
        model = Sequential([
            layers.Dense(15, input_shape=(7,), activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ])

        # Compile the model
        start = time.time()
        model.compile(loss=BinaryCrossentropy(), optimizer=optimizers.Adam(learning_rate=0.01))

        # Train the model
        model.fit(X_train, Y_train, epochs=1000, batch_size=100)
        time_n = (time.time()) - start

        Y_pred = model.predict(X_test)

        wrong = 101 - Y_pred[Y_pred > 0.5].shape[0]

        exp_results.append(np.array([time_n, wrong]))
    return exp_results


exp_ = run_experiment(3)
tf_exp_ = run_experiment_tf(3)
np_exp_ = np.array(exp_)
np_exp_tf = np.array(tf_exp_)


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot 1: Runtime
axes[0].plot(np_exp_[:, 0], label="Our")
axes[0].plot(np_exp_tf[:, 0], label="Tensorflow")
axes[0].set_xticks(np.arange(0, len(exp_), 1.0))
axes[0].legend()
axes[0].set_title("Runtime for 1000 epochs, 0.01 alpha, 100 batch_size")
axes[0].set_xlabel("Experiment #")
axes[0].set_ylabel("Runtime (in seconds)")

# Plot 2: Wrong Predictions
axes[1].plot(np_exp_[:, 1], label="Our")
axes[1].plot(np_exp_tf[:, 1], label="Tensorflow")
axes[1].set_xticks(np.arange(0, len(exp_), 1.0))
axes[1].legend()
axes[1].set_title("Wrong predictions, 1000 epochs, 0.01 alpha, 100 batch_size")
axes[1].set_xlabel("Experiment #")
axes[1].set_ylabel("Wrong predictions")

# Plot 3: Mean Error
mean_err_np = np.mean(np_exp_[:, 1])
mean_err_tf = np.mean(np_exp_tf[:, 1])
axes[2].bar(['Our', 'Tensorflow'], [mean_err_np, mean_err_tf])
axes[2].set_title("Mean error (examples misclassified)")
axes[2].set_ylabel("Mean error")
axes[2].set_ylim([0, np.max([mean_err_np, mean_err_tf]) * 1.1])

plt.tight_layout()
plt.show()
# TF 26.19999999999999