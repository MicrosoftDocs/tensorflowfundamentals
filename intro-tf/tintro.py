# Additional functions for Microsoft Learn Module
# Introduction to TensorFlow

import gzip
import numpy as np
import tensorflow as tf
from typing import Tuple


class NeuralNetwork(tf.keras.Model):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    initializer = tf.keras.initializers.GlorotUniform()
    self.W1 = tf.Variable(initializer(shape=(784, 20)))
    self.b1 = tf.Variable(tf.zeros(shape=(20,)))
    self.W2 = tf.Variable(initializer(shape=(20, 10)))
    self.b2 = tf.Variable(tf.zeros(shape=(10,)))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    x = tf.reshape(x, [-1, 784])
    x = tf.matmul(x, self.W1) + self.b1
    x = tf.nn.relu(x)
    x = tf.matmul(x, self.W2) + self.b2
    return x  

def read_images(path: str, image_size: int, num_items: int) -> np.ndarray:
  with gzip.open(path, 'rb') as file:
    data = np.frombuffer(file.read(), np.uint8, offset=16)
    data = data.reshape(num_items, image_size, image_size)
  return data

def read_labels(path: str, num_items: int) -> np.ndarray:
  with gzip.open(path, 'rb') as file:
    data = np.frombuffer(file.read(num_items + 8), np.uint8, offset=8)
    data = data.astype(np.int64)
  return data

def get_data(batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  image_size = 28
  num_train = 60000
  num_test = 10000

  training_images = read_images('data/FashionMNIST/raw/train-images-idx3-ubyte.gz', image_size, num_train)
  test_images = read_images('data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz', image_size, num_test)
  training_labels = read_labels('data/FashionMNIST/raw/train-labels-idx1-ubyte.gz', num_train)
  test_labels = read_labels('data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz', num_test)

  # (training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

  train_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
  test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

  train_dataset = train_dataset.map(lambda image, label: (float(image) / 255.0, label))
  test_dataset = test_dataset.map(lambda image, label: (float(image) / 255.0, label))

  train_dataset = train_dataset.batch(batch_size).shuffle(500)
  test_dataset = test_dataset.batch(batch_size).shuffle(500)

  return (train_dataset, test_dataset)


labels_map = {
  0: 'T-Shirt',
  1: 'Trouser',
  2: 'Pullover',
  3: 'Dress',
  4: 'Coat',
  5: 'Sandal',
  6: 'Shirt',
  7: 'Sneaker',
  8: 'Bag',
  9: 'Ankle Boot',
}
