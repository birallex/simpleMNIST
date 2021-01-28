from keras.datasets import mnist
from simple_mnist_model import build_model
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255
test_labels = to_categorical(test_labels)

simple_mnist_model = build_model()
simple_mnist_model.load_weights('simple_mnist.h5')

loss, accuracy = simple_mnist_model.evaluate(test_images, test_labels)
print('test_accuracy: ', accuracy)