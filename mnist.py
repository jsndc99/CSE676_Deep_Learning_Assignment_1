import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 784)) / 255.0
test_images = test_images.reshape((10000, 784)) / 255.0

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,), name='hidden_layer'),
    layers.Dense(10, activation='softmax', name='output_layer')
], name='mnist_fc_model')

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

summary_writer = tf.summary.create_file_writer('logs')
with summary_writer.as_default():
    tf.summary.trace_on(graph=True, profiler=True)
    model(tf.zeros((1, 784)))
    tf.summary.trace_export(name='mnist_fc_model_graph', step=0,profiler_outdir='logs')

batch_size = 64
num_epochs = 10
num_batches = train_images.shape[0] // batch_size

for epoch in range(num_epochs):
    for batch in range(num_batches):

        batch_images = train_images[batch*batch_size : (batch+1)*batch_size]
        batch_labels = train_labels[batch*batch_size : (batch+1)*batch_size]

        loss, accuracy = model.train_on_batch(batch_images, batch_labels)

        step = epoch * num_batches + batch
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', loss, step=step)
            tf.summary.scalar('train_accuracy', accuracy, step=step)

    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    with summary_writer.as_default():
        tf.summary.scalar('test_loss', test_loss, step=(epoch+1)*num_batches)
        tf.summary.scalar('test_accuracy', test_accuracy, step=(epoch+1)*num_batches)

model.save('mnist_fc_model.h5')