# -*- coding: utf-8 -*-
# Author : Arthur CARON, largely inspired from https://www.tensorflow.org/tutorials/images/classification
# This code trains the last layers of a network which aims at detecting cats and dogs. The link for the dataset is line 21.
# It generate a tflite file of the network and converts it to a .cpp file so that the weights of the network can easily
# integrated in Arduino IDE for our ESP32 board.
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import random as rd
import cv2 as cv
import zipfile

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# os.system("wget --no-check-certificate 
#   https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip 
#   -O /home/arthur/Téléchargements/cats_and_dogs.zip")

train_dir = pathlib.Path('/home/arthur/Documents/cats_and_dogs/train')
image_count = len(list(train_dir.glob('*/*.jpg')))
print("There are {} total images:".format(image_count))

cats = list(train_dir.glob('cats/*'))
PIL.Image.open(str(cats[0]))
PIL.Image.open(str(cats[1]))

dogs = list(train_dir.glob('dogs/*'))
PIL.Image.open(str(dogs[0]))
PIL.Image.open(str(dogs[1]))

batch_size = 32
img_height = 64
img_width = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

"""You can find the class names in the `class_names` attribute on these datasets. These correspond to the directory names in alphabetical order."""

class_names = train_ds.class_names
print(class_names)

"""## Visualize the data

Here are the first images from the training dataset.

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

You will train a model using these datasets by passing them to `model.fit` in a moment. If you like, you can also manually iterate over the dataset and retrieve batches of images:"""

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

"""## Standardize the data

The RGB channel values are in the `[0, 255]` range. This is not ideal for a neural network; in general you should seek to make your input values small. Here, you will standardize values to be in the `[0, 1]` range by using a Rescaling layer.
"""

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

"""Note: The Keras Preprocessing utilities and layers introduced in this section are currently experimental and may change.

There are two ways to use this layer. You can apply it to the dataset by calling map:
"""

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

"""Or, you can include the layer inside your model definition, which can simplify deployment. Let's use the second approach here.

Note: you previously resized images using the `image_size` argument of `image_dataset_from_directory`. If you want to include the resizing logic in your model as well, you can use the [Resizing](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Resizing) layer.

# Create the model

The model consists of three convolution blocks with a max pool layer in each of them. There's a fully connected layer with 128 units on top of it that is activated by a `relu` activation function. This model has not been tuned for high accuracy, the goal of this tutorial is to show a standard approach.
"""

num_classes = 2

"""
## Overfitting
## Data augmentation
"""

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

"""
## Dropout
# """

model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

"""## Compile and train the model"""

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 15
history = model.fit(
  train_ds,
  validation_data=valid_ds,
  epochs=epochs
)

converter = tf.lite.TFLiteConverter.from_keras_model(model) 
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
os.system ("xxd -i model.tflite > model.cpp")


"""## Visualize training results

After applying data augmentation and Dropout, there is less overfitting than before, and training and validation accuracy are closer aligned. 
"""

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Predict on new data

test_dir = pathlib.Path('/home/arthur/Documents/cats_and_dogs/test')
image_count = len(list(test_dir.glob('*/*.jpg')))
print("There are {} test images:".format(image_count))

# cellphone_to_test = list(train_dir.glob('./*'))
# PIL.Image.open(str(cellphone_to_test[0]))


def capture_from_webcam():
  capture = cv.VideoCapture(0)
  while True:
      isTrue,frame = capture.read()
      frame = cv.resize(frame,(64,64))
      img_array = keras.preprocessing.image.img_to_array(frame)
      img_array = tf.expand_dims(img_array, 0) # Create a batch

      predictions = model.predict(img_array)
      score = tf.nn.softmax(predictions[0])

      print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
      )

      cv.imshow('Video',frame)
      if cv.waitKey(20) & 0xFF==ord('q'):
          break
  capture.release()
  cv.destroyAllWindows()

def capture_from_folder():
  test_path = "/home/arthur/Documents/cats_and_dogs/test"
  files = os.listdir(test_path)
  i = 1
  count = 0

  for f in files:
    i+=1
    target = test_path+'/'+str(f)

    img = keras.preprocessing.image.load_img(
        target, target_size=(img_height, img_width)
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # print(
    #     "This image {} most likely belongs to {} with a {:.2f} percent confidence."
    #     .format(f,class_names[np.argmax(score)], 100 * np.max(score))
    # )
    if f[:3] == class_names[np.argmax(score)][:3]:
      count+=1
      print(f,"valid",100 * np.max(score))
    else:
      print(f,"--echec",100 * np.max(score))

  print("taux de reussite : {} reussites sur {} tests, soit {:.2f} %".format(count,i,count/i))

# capture_from_webcam()
capture_from_folder()