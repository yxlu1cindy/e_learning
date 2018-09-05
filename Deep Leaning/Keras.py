Cited From https://keras.io/getting-started/sequential-model-guide/

## Build a Sequential Model
from keras.models import Sequential
from keras.layers import Dense, Activation

# Create the model
model = Sequential()

# Add Layers
## in the first layer, must specify the expected input data shape:
##M2
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
##M2
model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# Learning Process
## For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)  ##Convert labels to categorical one-hot encoding

## For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

## For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

## For custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

#Model shape
model.output_shape
# Model summary
model.summary()

# Model config
model.get_config()

# List all weight tensors 
model.get_weights()


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
              
# Training
## x_train and y_train are Numpy arrays--same as sklearns
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluation
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

#Prediction
classes = model.predict(x_test, batch_size=128)

# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
confusion_matrix(y_test, y_pred)
# Precision 
precision_score(y_test, y_pred)
# Recall  a measure of a classifier’s completeness. The higher the recall, the more cases the classifier covers.
recall_score(y_test, y_pred)
# F1 score  a weighted average of precision and recall.
f1_score(y_test,y_pred)
# Cohen's kappa   the classification accuracy normalized by the imbalance of the classes in the data.
cohen_kappa_score(y_test, y_pred)

from sklearn.metrics import r2_score
##besides the MSE and MAE scores, you could also use the R2 score or the regression score function.
