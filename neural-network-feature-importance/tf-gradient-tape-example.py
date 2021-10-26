import numpy as np
import tensorflow as tf
# Fetch and format the mnist data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()


# custom_sample_weight
uniq_labels, cnts = np.unique(mnist_labels,return_counts=True)

cnts = cnts/np.sum(cnts)

custom_sample_weight_lkup = dict(zip(uniq_labels, cnts))

sample_weight = np.array(list(map(lambda x: custom_sample_weight_lkup[x], mnist_labels)))


dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels,tf.int64),
   tf.cast(sample_weight,tf.float32))
    )
dataset = dataset.shuffle(1000).batch(32)

# Build the model
mnist_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                         input_shape=(None, None, 1)),
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])


loss_func = tf.keras.losses.SparseCategoricalCrossentropy(
                                              from_logits=True,reduction=tf.keras.losses.Reduction.NONE)

@tf.function()
def custom_loss_object(ytrue,ypred,sample_weight):
    loss = loss_func(ytrue,ypred)
    loss = tf.multiply(loss,sample_weight)
    return tf.reduce_mean(loss)



loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

# The loss function to be optimized


loss_history = []


def train_step(images, labels, sw):
    
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)
    
        # Add asserts to check the shape of the output.
        tf.debugging.assert_equal(logits.shape, (32, 10))

        loss_value = custom_loss_object(labels,logits,sw)

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))


epochs=3

for epoch in range(epochs):
    for (batch, (images, labels,sw)) in enumerate(dataset):
        train_step(images, labels,sw)
    print ('Epoch {} finished'.format(epoch))
    
for images,labels,sw in dataset.take(1):
    print("Logits: ", custom_loss_object(labels[0:1],mnist_model(images[0:1]),sw))
    
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
