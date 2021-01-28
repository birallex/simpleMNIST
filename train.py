from keras.datasets import mnist
from simple_mnist_model import build_model
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

simple_mnist_model = build_model()
simple_mnist_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') /255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

history = simple_mnist_model.fit(train_images, train_labels, epochs = 5, batch_size = 128)
simple_mnist_model.save_weights('simple_mnist.h5')

print(history.history)
loss, accuracy = simple_mnist_model.evaluate(test_images, test_labels)
print('test_accuracy: ', accuracy)