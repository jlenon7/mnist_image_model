from constants import MODEL_NAME
import tensorflow.keras as keras

from os.path import exists
from typing import Optional

class Path:
  def plots(self, path: Optional[str]):
    return self.storage(f'plots/{path}')

  def board(self, path: Optional[str]):    
    return self.storage(f'board/{path}')

  def logs(self, path: Optional[str]):    
    return self.storage(f'logs/{path}')

  def storage(self, path: Optional[str]):    
    path = self.clean_path(path) 

    return f'storage{path}'

  def resources(self, path: Optional[str]):    
    path = self.clean_path(path) 

    return f'resources{path}'

  def clean_path(self, path: Optional[str]):
    if path is None:
      return ''

    if path.endswith('/') is True:
      path = path[:-1]

    if path.startswith('/') is True:
      return path 

    return f'/{path}'

path = Path()

def load_model():
  path = f'storage/{MODEL_NAME}'
  model_exists = exists(path)

  if (model_exists):
    return keras.models.load_model(path)

  model = keras.models.Sequential()

  model.add(
    keras.layers.Conv2D(
      filters=32,
      kernel_size=(4, 4),
      input_shape=(28, 28, 1),
      activation='relu'
    )
  )
  model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(128, activation='relu'))
  model.add(keras.layers.Dense(10, activation='softmax'))

  model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
  )

  return model

def get_df(): 
  return keras.datasets.mnist.load_data()
