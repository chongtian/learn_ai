from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# function to load data from CSV file
def get_dataset(file_path, label_column, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=50, 
      label_name=label_column,
      na_value="?",
      num_epochs=1,
      ignore_errors=True, 
      **kwargs)
  return dataset

def pack(features, label):
  return tf.stack(list(features.values()), axis=-1), label

# import model data from csv
train_file_path = 'VUG-data.csv'
predict_file_path = 'VUG-test.csv'

# Get CSV_COLUMNS and DEFAULTS
# These shall match with the CSV files
CSV_COLUMNS = []
DEFAULTS = []
for i in range(305):
    label = "d" + str(i).zfill(3)
    CSV_COLUMNS.append(label)
    DEFAULTS.append(0.0)
CSV_COLUMNS.append('signal')
DEFAULTS.append(0)
    
# define LABEL_CONLUMN
LABEL_COLUMN = 'signal'

# Get Data set from csv and pre-processing
raw_data = get_dataset(train_file_path, LABEL_COLUMN, column_names=CSV_COLUMNS,  column_defaults = DEFAULTS)
packed_data = raw_data.map(pack)
test_data = packed_data.take(1)
train_data = packed_data.skip(1)

# Build the model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(305, activation='relu'),
  tf.keras.layers.Dense(1000, activation='relu'),
  tf.keras.layers.Dense(500, activation='relu'),
  tf.keras.layers.Dense(3, activation='softmax'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# Train Model
model.fit(train_data, epochs=5)

# evaluate model
test_loss, test_acc = model.evaluate(test_data)
print('\nTest accuracy:', test_acc)

# # Predict
# predictions = model.predict(test_data)
# # Show results
# for prediction, actual in zip(predictions, list(test_data)[0][1]):
#   print("Predicted signal: ", np.argmax(prediction), " Actual signal: ", actual.numpy())

# Do actual prediction
print("\nPredict data from file ", predict_file_path)
raw_data = get_dataset(predict_file_path, LABEL_COLUMN, column_names=CSV_COLUMNS,  column_defaults = DEFAULTS)
packed_data = raw_data.map(pack)
predictions = model.predict(packed_data)
for prediction in predictions:
  print("Predicted signal: ", np.argmax(prediction))