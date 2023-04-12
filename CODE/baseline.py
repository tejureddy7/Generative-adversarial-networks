
"""

This section will import the packages necessary to run the code and identify the paths of the training, testing, and the prediction datasets.
"""

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_preprocessing import image
import pandas as pd
import tqdm
from time import time
import pandas as pd
import seaborn as sns
import random
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from PIL import Image, ImageFile

i
ImageFile.LOAD_TRUNCATED_IMAGES = True
start_time = time()
batch_size = 32
sns.set_style('darkgrid')

"""Paths for the training dataset, testing dataset, and the prediction dataset.
Step1: Unzip archive.zip
To use the original dataset unzip the archive.zip file and use Train and Test files

TEST_FILES = "archive/seg_test/seg_test"
TRAIN_FILES = "archive/seg_train/seg_train"

To use the original Dataset with augmented dataset unzip Full.zip into fulldata
folder and use
TEST_FILES = "archive/seg_test/seg_test" #validation set
TRAIN_FILES = "fulldata/Full/seg_train"

To train with just the augmented dataset unzip ganseg_combined.zip into 2gancombined
folder and use
TEST_FILES = "archive/seg_test/seg_test" #validation set
TRAIN_FILES = "f2gancombined/ganseg_combined/seg_train"

"""

#for now lets use this
TEST_FILES = "archive/seg_test/seg_test" #validation set
TRAIN_FILES = "fulldata/Full/seg_train"


"""## Model definition (MANDATORY)

This section defines the VGG19 in the function  `vgg19_model_generator()`. The function can be called and the returned model can be stored into a variable.

Defines the VGG19 model.
"""

def vgg19_model_generator():
    base_model = VGG19(
        weights = "imagenet",
        input_shape = (64, 64, 3),
        include_top = False)

    for layers in base_model.layers:
        layers.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(100, activation = "relu"),
        Dropout(0.4),
        Dense(64, activation = "relu"),
        Dense(6, activation = "softmax")
      ])

    return model

tf.keras.backend.clear_session()

"""## Generating the training and validation DataGenerator objects (MANDATORY)

Creating ImageDataGenerator objects for training and testing datasets.
"""

train_datagen = ImageDataGenerator(
    # rotation_range = 15,
    # horizontal_flip = True,
    preprocessing_function = preprocess_input
)

val_datagen = ImageDataGenerator(
    # rotation_range = 15,
    # horizontal_flip = True,
    preprocessing_function = preprocess_input
)

test_datagen = ImageDataGenerator(
    # rotation_range = 15,
    # horizontal_flip = True,
    preprocessing_function = preprocess_input
)





"""Creates a dataframe to map image to class for the validation dataset."""
images = []
labels =[]
train_data_dir= TRAIN_FILES
val_dat_dir = TEST_FILES
for sub_dir in os.listdir(train_data_dir):
  image_list=os.listdir(os.path.join(train_data_dir,sub_dir))  #list of all image names in the directory
  image_list = list(map(lambda x:os.path.join(sub_dir,x),image_list))
  images.extend(image_list)
  labels.extend([sub_dir]*len(image_list))

print(type(images))
print(type(labels))
df_train = pd.DataFrame({"Images":images,"Labels":labels})

"""Creates a dataframe to map image to class for the validation dataset."""
images = []
labels =[]
for sub_dir in os.listdir(val_dat_dir):

  image_list=os.listdir(os.path.join(val_dat_dir,sub_dir))  #list of all image names in the directory
  image_list = list(map(lambda x:os.path.join(sub_dir,x),image_list))
  images.extend(image_list)
  labels.extend([sub_dir]*len(image_list))


df_val = pd.DataFrame({"Images":images,"Labels":labels})
print("shape",df_val.shape)

"""split validation dataset into validation and test set for ease of use"""
import pandas as pd

# Shuffle your dataset
shuffle_df = df_val.sample(frac=1)

# Define a size for your train set
train_size = int(0.7 * len(df_val))

# Split your dataset
df_val= shuffle_df[:train_size]
df_test = shuffle_df[train_size:]
print("shapee",df_val.shape)



"""Creating training and validation generators for the training and validation datasets respectively."""

train_generator=train_datagen.flow_from_dataframe(
dataframe=df_train,
directory=train_data_dir,
x_col="Images",
y_col="Labels",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(64,64))

val_generator=val_datagen.flow_from_dataframe(
dataframe=df_val,
directory=val_dat_dir,
x_col="Images",
y_col="Labels",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(64,64))

test_generator=val_datagen.flow_from_dataframe(
dataframe=df_test,
directory=val_dat_dir,
x_col="Images",
y_col="Labels",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(64,64))

"""Test data"""

test_generator.reset()
x_test, y_test = next(test_generator)
for i in tqdm.tqdm(range(int(test_generator.n/32)-1)):
  img, label = next(test_generator)
  x_test = np.append(x_test, img, axis=0 )
  y_test = np.append(y_test, label, axis=0)
print(x_test.shape, y_test.shape)

"""## Checkpoints and Model Compilation and Training (MANDATORY)

Creating checkpoints and defining learning rate reduction conditions.
"""

reduce_lr = ReduceLROnPlateau(
    monitor = "val_accuracy",
    patience = 2,
    verbose = 1,
    factor = 0.5,
    min_lr = 0.000000001
)

early_stopping = EarlyStopping(
    monitor = "val_accuracy",
    patience = 10,
    verbose = 1,
    mode = "max",
)

checkpoint = ModelCheckpoint(
    monitor = "val_accuracy",
    filepath = "intel_img_class_vgg16.{epoch:02d}-{val_accuracy:.6f}.hdf5",
    verbose = 1,
    save_best_only = True,
    save_weights_only = True
)

"""Generating and compiling the model."""

model = vgg19_model_generator()
model_start = time()
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=['accuracy',tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.TruePositives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalseNegatives()])

model.summary()

"""Fitting the model on the training data and validating with the validation data at every epoch."""

history = model.fit(
    train_generator,
    epochs = 30,
    validation_data = val_generator
)



model.evaluate(x_test, y_test)
model_end = time()
model.save('justgancombined.h5')

"""## Accuracy metrics graphed (MANDATORY)

Graphing the accuracy metrics of the model throughout the training process.
"""

fig, axes = plt.subplots(1, 2, figsize = (24, 8))

sns.lineplot(x = range(len(history.history["loss"])), y = history.history["loss"], ax = axes[0], label = "Training Loss")
sns.lineplot(x = range(len(history.history["loss"])), y = history.history["val_loss"], ax = axes[0], label = "Validation Loss")

sns.lineplot(x = range(len(history.history["accuracy"])), y = history.history["accuracy"], ax = axes[1], label = "Training Accuracy")
sns.lineplot(x = range(len(history.history["accuracy"])), y = history.history["val_accuracy"], ax = axes[1], label = "Validation Accuracy")
axes[0].set_title("Loss"); axes[1].set_title("Accuracy")

sns.despine()
plt.show()
fig.savefig("justgancombines.png")


print(f"Model execution time = {model_end - model_start}")
print(f"Overall execution = {time() - start_time}")
