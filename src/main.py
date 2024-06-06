import time
import calendar
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import MODEL_NAME
from helpers import path, get_df, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix

(X_train, y_train), (X_test, y_test) = get_df()

y_cat_test = to_categorical(y_test, num_classes=10)
y_cat_train = to_categorical(y_train, 10)

X_test = X_test / 255
X_train = X_train / 255

# batch_size, width, height, color_channels
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.reshape(60000, 28, 28, 1)

model = load_model()

model.fit(
  X_train,
  y_cat_train,
  epochs=10,
  validation_data=(X_test, y_cat_test),
  callbacks=[
    EarlyStopping(monitor='val_loss', patience=1),
    TensorBoard(
      log_dir=path.board(f'fit-{calendar.timegm(time.gmtime())}'),
      histogram_freq=1,
      write_graph=True,
      write_images=True,
      update_freq='epoch',
      profile_batch=2,
      embeddings_freq=1
    )
  ]
)

metrics = pd.DataFrame(model.history.history)

plt.figure(figsize=(10,8))
metrics[['loss', 'val_loss']].plot()
plt.savefig(path.plots('loss.png'))

plt.figure(figsize=(10,8))
metrics[['accuracy', 'val_accuracy']].plot()
plt.savefig(path.plots('accuracy.png'))

# For binary classification
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)

confusion_matrix_result = confusion_matrix(y_test, predictions)
classification_report_result = classification_report(y_test, predictions)

print()
print('Confusion Matrix:')
print(confusion_matrix_result)
print()
print('Classification Report:')
print(classification_report_result)

plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix_result, annot=True)
plt.savefig(path.plots('heatmap-confusion-matrix.png'))

print()
print('Saving the model at', path.storage(MODEL_NAME))
model.save(path.storage(MODEL_NAME))
