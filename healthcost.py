!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from IPython.display import clear_output

print(tf.__version__)

# Import data
dataset_path = keras.utils.get_file("insurance.csv", "https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv")
dataset = pd.read_csv(dataset_path)
dataset.head()

#checking data
dataset.isna().sum()

dataset['sex'] = dataset['sex'].apply(lambda x: 1 if x=='male' else 0 )
dataset['smoker'] = dataset['smoker'].apply(lambda x: 1 if x=='yes' else 0 )
dataset['region'] = dataset['region'].apply(lambda x: 1 if x=='southwest' else (2 if x=='southeast' else ( 3 if  x=='northwest' else 4 )) )
dataset.head()

dataset['region'] = dataset['region'].map({1: 'southwest', 2: 'southeast', 3: 'northwest', 4: 'northeast'})

dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
dataset.tail()

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[["age", "expenses", "bmi", "children"]], diag_kind="kde")

#getting train stats
train_stats = train_dataset.describe()
train_stats.pop("expenses")
train_stats = train_stats.transpose()
train_stats

train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
  
model = build_model()


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(train_dataset, train_labels, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])
                    
#testing modell                    
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
                    
