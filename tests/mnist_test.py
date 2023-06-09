import tensorflow as tf

tf.keras.utils.disable_interactive_logging()

from keras import layers, Sequential
from matplotlib import pyplot as plt
import numpy as np

from cuda.cuda_NN import NN, Layer
from cuda.cuda_optimizers import Adam
import time

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.

x_train_flattened = x_train.reshape(x_train.shape[0], -1)
x_test_flattened = x_test.reshape(x_test.shape[0], -1)

y_train_oh = tf.keras.utils.to_categorical(y_train)
y_test_oh = tf.keras.utils.to_categorical(y_test)

start = time.time() * 1000


def run_experiment(n):
    # np.argmax(Y_pred, axis=1)
    exp_results = []
    for i in range(n):
        net = NN(layers=[
            Layer(784, 512, activation='relu'),
            Layer(512, 128, activation='relu'),
            Layer(128, 64, activation='relu'),
            Layer(64, 10, activation='softmax'),
        ], loss='categorical_crossentropy', optimizer=Adam())
        start = time.time()
        net.fit(x_train_flattened, y_train_oh, 10, 0.01, sample_size=1000, include_logs=True)
        time_n = (time.time()) - start
        Y_pred = net.predict(x_test_flattened)

        Y_pred_oh = np.argmax(Y_pred, axis=1)
        wrong = 0
        for i in range(Y_pred_oh.shape[0]):
            pred = Y_pred_oh[i]
            actual = y_test[i]
            if pred != actual:
                wrong += 1

        exp_results.append(np.array([time_n, wrong]))
    return exp_results


def run_experiment_tf(n):
    # np.argmax(Y_pred, axis=1)
    exp_results = []
    for i in range(n):
        # Create the model using the Sequential API
        model = Sequential([
            layers.Dense(512, input_shape=(784,), activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        # Compile the model
        start = time.time()
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

        # Train the model
        model.fit(x_train_flattened, y_train_oh, epochs=10, batch_size=1000)
        time_n = (time.time()) - start

        Y_pred = model.predict(x_test_flattened)

        Y_pred_oh = np.argmax(Y_pred, axis=1)
        wrong = 0
        for i in range(Y_pred_oh.shape[0]):
            pred = Y_pred_oh[i]
            actual = y_test[i]
            if pred != actual:
                wrong += 1
        exp_results.append(np.array([time_n, wrong]))
    return exp_results


exp_ = run_experiment(1)
np_exp_ = np.array(exp_)
exp_tf = run_experiment_tf(1)
np_exp_tf = np.array(exp_tf)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot 1: Runtime
axes[0].plot(np_exp_[:, 0], label="Our")
axes[0].plot(np_exp_tf[:, 0], label="Tensorflow")
axes[0].set_xticks(np.arange(0, len(exp_), 1.0))
axes[0].legend()
axes[0].set_title("Runtime for 10 epochs, 0.01 alpha, 1000 batch_size")
axes[0].set_xlabel("Experiment #")
axes[0].set_ylabel("Runtime (in seconds)")

# Plot 2: Wrong Predictions
axes[1].plot(np_exp_[:, 1], label="Our")
axes[1].plot(np_exp_tf[:, 1], label="Tensorflow")
axes[1].set_xticks(np.arange(0, len(exp_), 1.0))
axes[1].legend()
axes[1].set_title("Wrong predictions, 10 epochs, 0.01 alpha, 1000 batch_size")
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
