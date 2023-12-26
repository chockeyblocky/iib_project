"""
This will test the SSIM loss function.
"""

from losses.ssim import SSIM
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from layers.graph_transformer import GraphTransformer

class Net(tf.keras.Model):
    def __init__(self, edge_n):
        super().__init__()
        self.gt = GraphTransformer(
                  depth = 3,
                  heads = 4,
                  edge_dim = edge_n,
                  with_feedforwards = True,
                  rel_pos_emb = True)
        self.dense = tf.keras.layers.Dense(3, activation='softplus')
        # add in shape reduction layers to make shapes match

    def call(self, nodes, edges, mask):
        x, edges_new = self.gt(nodes, edges, mask=mask)
        x = self.dense(x)
        return x

model = Net(edge_n=5)

ssim = SSIM(k1=0.01, k2=0.03, L=255)

inputs = tf.keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(64, activation="relu")(x1)
x2 = layers.Dense(1024, name="predictions", activation='softplus')(x2)
outputs = layers.Reshape([32, 32, 1])(x2)
# model = keras.Model(inputs=inputs, outputs=outputs)

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
# Instantiate a loss function.
loss_fn = tf.keras.losses.MeanAbsoluteError()

# Prepare the training dataset.
batch_size = 1
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        n = 20
        nodes = tf.random.normal([batch_size, n, 27])
        edges = tf.random.normal([batch_size, n, n, 5])
        mask = tf.cast(tf.ones([batch_size, n]), tf.bool)
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            y = model(nodes, edges, mask)

            # ERROR IS CAUSED BY THESE TWO LINES + SSIM
            y = tf.repeat(y, repeats=y.shape[1], axis=0)
            # PROGRAM L2 NORM MANUALLY
            y = tf.square(y - tf.transpose(y, perm=[1, 0, 2])) / 3

            y = tf.expand_dims(tf.expand_dims(tf.reduce_sum(y, axis=-1), 0), -1)

            # Compute the loss value for this minibatch.
            y_batch = tf.reshape(y_batch_train, [-1, 1, 1])
            y_batch = tf.expand_dims(y_batch, axis=-1)
            y_batch = tf.repeat(y_batch, n, axis=1)
            y_batch = tf.repeat(y_batch, n, axis=2)
            y_batch = tf.cast(y_batch, tf.float32)


            y = tf.cast(y, tf.float32)
            loss1 = tf.cast(tf.reduce_mean(tf.keras.losses.mean_absolute_error(y, y_batch)), tf.float32)
            loss2 = 1 - ssim.ssim_loss(img1=y, img2=y_batch)

            loss = (loss1 + 20 * loss2) / 4

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, model.trainable_weights)
        print(grads)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss))
            )
            print("Seen so far: %s samples" % ((step + 1) * batch_size))