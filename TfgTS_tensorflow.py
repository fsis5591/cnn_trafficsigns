from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import os
import random

import urllib
from six.moves import urllib

import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import cv2


import numpy as np
import tensorflow as tf

def load_data(data_dir):
    """Loads a data set and returns two lists:
    
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels









def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # images are 56x56 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 56, 56, 1])
  print("input layer:", format(input_layer.shape))

  # Convolutional Layer #1
  # Computes 100 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 56, 56, 1]
  # Output Tensor Shape: [batch_size, 56, 56, 100]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=100,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  print("conv1:", format(conv1.shape))    

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 56, 56, 100]
  # Output Tensor Shape: [batch_size, 28, 28, 100]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  print("pool1:", format(pool1.shape))      

  # Convolutional Layer #2
  # Computes 100 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 100]
  # Output Tensor Shape: [batch_size, 28, 28, 150]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=150,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  print("conv2:", format(conv2.shape)) 

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 150]
  # Output Tensor Shape: [batch_size, 14, 14, 150]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  print("pool2:", format(pool2.shape)) 

  # Convolutional Layer #3
  # Computes 150 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 150]
  # Output Tensor Shape: [batch_size, 14, 14, 250]
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=250,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  print("conv3:", format(conv3.shape))    

  # Pooling Layer #3
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 250]
  # Output Tensor Shape: [batch_size, 7, 7, 250]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  print("pool3:", format(pool3.shape))    

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 250]
  # Output Tensor Shape: [batch_size, 7 * 7 * 250]
  pool3_flat = tf.reshape(pool3, [-1, 7 * 7 * 250])
  print("pool3_flat:", format(pool3_flat.shape))   

  # Dense Layer#1
  # Densely connected layer with 521 neurons
  # Input Tensor Shape: [batch_size,  7 * 7 * 250]
  # Output Tensor Shape: [batch_size, 512]
  dense1 = tf.layers.dense(inputs=pool3_flat, units=512, activation=tf.nn.relu)
  print("dense1:", format(dense1.shape))  

  # Add dropout operation; 0.6 probability that element will be kept
  dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  print("dropout1:", format(dropout1.shape))     

  # Dense Layer#2
  # Densely connected layer with 256 neurons
  # Input Tensor Shape: [batch_size,  512]
  # Output Tensor Shape: [batch_size, 256]
  dense2 = tf.layers.dense(inputs=dropout1, units=256, activation=tf.nn.relu)
  print("dense2:", format(dense2.shape))     

#  # Add dropout operation; 0.6 probability that element will be kept
#  dropout2 = tf.layers.dropout(
#      inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 256]
  # Output Tensor Shape: [batch_size, 62]
  logits = tf.layers.dense(inputs=dense2, units=62)
  print("logits:", format(logits.shape))  


  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=62)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




def main(unused_argv):   
    
  #Load datasets.
#  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#      filename=DATA_TRAINING,
#      target_dtype=np.int,
#      features_dtype=np.float32)
#  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#      filename=DATA_TEST,
#      target_dtype=np.int,
#      features_dtype=np.float32)
#  predict_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#      filename=DATA_PREDICT,
#     target_dtype=np.int,
#      features_dtype=np.float32)  


  # Load training and testing datasets.
  ROOT_PATH = "/home/felipe/traffic-signs-tensorflow-master/traffic"
  train_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Training")
  test_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Testing")

  # Load the training dataset.
  images, labels = load_data(train_data_dir)
  train_labels = np.asarray(labels, dtype=np.int32)

  # Load the test dataset.
  test_images, test_labels = load_data(test_data_dir)
  test1_labels = np.asarray(test_labels, dtype=np.int32)
  
    
  #### Resize Train images
  ##resize and normalize values to range 0.0->1.0
  #images32 = [skimage.transform.resize(image, (32, 32))
  #               for image in images]

  ##resize without normalizing keeping range 0->255
  images56 = [cv2.resize(image, (56,56))
                for image in images]

  ##->input shape (56,56,3)
  images_56gray = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images56]
  ##->output shape (56,56,1)
  train_images_56gray= np.asarray(images_56gray, dtype=np.float32)


  #### Resize Test images
  ##resize without normalizing keeping range 0->255
  test_images56 = [cv2.resize(image, (56,56))
                for image in test_images]

  ##->input shape (56,56,3)
  test_images_56gray = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in test_images56]
  ##->output shape (56,56,1)
  test1_images_56gray= np.asarray(test_images_56gray, dtype=np.float32)
    
    
    
  # Create the Estimator
  placas_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/home/felipe/traffic-signs-tensorflow-master/estimator")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_images_56gray},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  placas_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test1_images_56gray},
      y=test1_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = placas_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

#  # Print out predictions
#  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
#      x={"x": np.array(predict_set.data)},
#      num_epochs=1,
#      shuffle=False)
#  predictions = placas_classifier.predict(input_fn=predict_input_fn)
#  for i, p in enumerate(predictions):
#    print("Prediction %s: %s" % (i + 1, p))


if __name__ == "__main__":
  tf.app.run()





