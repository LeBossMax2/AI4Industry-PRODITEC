import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
import shutil

class Ai :
  
  def train():

    global train_ds
    global val_ds
    global class_names

    data_dir = "data/IA" # modifier data pour n'avoir que deux categorie/fichier
    data_dir = pathlib.Path(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(480, 640),
      batch_size=32)

    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(480, 640),
      batch_size=32)

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    model = Sequential([
      layers.Rescaling(255, input_shape=(480, 640, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    epochs = 6
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
    )

    print('\nSaving...')
    model.save("AI4Indistry-PRODITEC/My_model")
    print('Ok')

  def predict(path) :

    model = keras.models.load_model("AI4Indistry-PRODITEC/My_model")

    data_test = pathlib.Path(path)
    
    img = tf.keras.utils.load_img(
        data_test 
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

#Ai.train()

Ai.predict("AI4Indistry-PRODITEC/data/IA/CAT2JPG/CAM1-2020-12-09 11-41-41-749.JPG")
  