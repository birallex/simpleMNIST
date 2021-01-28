from keras import models
from keras import layers

def build_model():
    simple_mnist_model = models.Sequential()
    simple_mnist_model.add(layers.Dense(512, activation = 'relu', input_shape = (28*28, )))
    simple_mnist_model.add(layers.Dense(10, activation = 'softmax'))
    simple_mnist_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])
    print(simple_mnist_model.summary())
    return simple_mnist_model