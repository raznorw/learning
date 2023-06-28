#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 23:33:38 2023

https://www.tensorflow.org/tutorials/keras/text_classification

Conversion of the basic text classification keras/tensorflow example
to do multi-class classification on the provided stack overflow dataset
"""

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

print (tf.__version__)

# Sentiment Analysis - Classify IMDB reviews as positive or negative

url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"

# NOTE: this was downloaded to /tmp/.keras, NOT $HOME/.keras  Unsure why
# This was also extracted as /tmp/.keras/{test,train}  no other structure was
# in the dataset
# cache_dir doesn't seem to do anything
dataset = tf.keras.utils.get_file("stack_overflow_16k", url,
                                  untar=True, cache_dir='data',
                                  cache_subdir='')

dataset_dir = os.path.dirname(dataset)
# list of contents of the specified directory
os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')
# training directory extracted from dataset contains neg and pos subdirs
# with the training samples
os.listdir(train_dir)
os.listdir(test_dir)

sample_file = os.path.join(train_dir, 'python/1021.txt')
with open(sample_file) as f:
    print(f.read())
    
    
# Now we need to prepare and load the dataset for training
# for binary classification, we expect two folders corresponding to the two 
# classes.  In our case, aclImdb/train/{neg,pos} are those two folders

# dataset metavariables
batch_size = 32
seed = 42

# load the data into a keras dataset
# automatically shuffles the data and partitions 20% for validation
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

# iterate over dataset to print a few examples.  Unnecessary
# We can see that the reviews contain some HTML code still (<br />), and that
# label 0 is negative, label 1 is positive
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])
        

# Now we can also load our validation and test datasets
# validation is the same dir as training, but uses the validation subset
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)
        
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    test_dir,
    batch_size=batch_size)

# Standardize, tokenize and vectorize the dataset
#  Standardize - preprocess the text.  In our case, remove HTML elements
#  Tokenize    - split the string into tokens.  In our case, words via whitespace
#  Vectorize   - convert tokens to numbers to feed to neural net

# We construct these operations into a layer for our neural network so they
# will be identical during training and test.  Could also preprocess all of the
# data, but then the proprocessing steps aren't built into the model and must
# be run separately on future datasets as well.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, 
                                    '[%s]' % re.escape(string.punctuation), '')

# Now we can use our standardization function in a TextVectorization layer
max_features = 10000     # number of features to learn
sequence_length = 250    # input size

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Use the adapt method of the layer to map the words to integer labels for
# use in the model.
# First, make a text only dataset without labels (train_text)
train_text = raw_train_ds.map(lambda x, y: x) # convert BatchDataset to MapDataset
# then, map the strings in the dataset to integers representing them
# this modifies the vectorize_layer to populate the vocabulary shown below
vectorize_layer.adapt(train_text)

## Look at results of preprocessing
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# retrieve a batch of 32 reviews+labels from dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", first_label)
print("Vectorized review", vectorize_text(first_review, first_label))

# We can reverse lookup the words in the vectorize layer
print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# Now we can apply our vectorization to our datasets to get those numerical words
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# TUNE dataset for performance
# .cache() keeps data in memory after its loaded off disk
# .prefetch() overlaps data preprocessing and model execution while training
# attempting to prevent disk-access from being a bottleneck
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Finally, we create our model 
# For text classification, we would like the numbers representing words
# to have some form of meaning, rather than the random assignment we gave
# above.
# The Embedding layer serves this purpose (see other tutorial).  Essentially,
# it learns an embedding for each word, which is a mapping from its integer to
# a vector of values.  This allows words to look more or less like each other
# to better represent similar concepts, meanings, or whatever features the
# model decides represents "closeness".
# The GlobalAveragePooling layer produces a fixed-length output averaged over
# the input sequence allowing us to handle variable length inputs.
# Our discriminator is a Dense single node to produce a binary classification
embedding_dim = 16
model=tf.keras.Sequential([
    layers.Embedding(max_features +1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(4)])   # 4 classes, so use 4 output nodes
model.summary()

model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

# Evaluate the model - we got about 87% accuracy
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# plot the accuracy vs. loss over time 
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# Note: each plot component displayed in console, but plt.show() did nothing
# The plots tab showed each individual component, but also not the combined plot
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plot training and validation accuracy
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

# Save the model so we can import it later without training
# We applied vectorization manually above, but we want it to be part of
# our exported model

# Applying the vectorization on the CPU / outside the model might be 
# better for training, but then we want to include it in the model 
# for deployment / ease of use.
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
   ])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), 
    optimizer='adam', metrics=['accuracy']
)

# test it on raw test dataset.  Should be the same results we got above
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)