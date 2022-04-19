import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import minmax_scale # single column normalization
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras import optimizers
from sklearn.metrics import confusion_matrix
import tensorflow.keras.regularizers as regularizers
import tensorflow as tf

# Load train_data and test_data from two separate csv files.
df = pd.read_csv('C:\\Users\\Alphacat\\Desktop\\CA ICU Project\\train_under.csv')#, header=None)
# train_col_name = df.columns
ycol=df[['hospital_death']]
print(ycol.shape)
del df['hospital_death']
print(df.shape)


train_onehot = pd.get_dummies(df, prefix_sep='_')
train_set = pd.concat([ycol,train_onehot],axis=1,sort=False)
train_set = train_set.loc[:, ~train_set.columns.duplicated()]
print(train_set.shape)
train_data = train_set.values
train_col_names = list(train_set.columns.values)
print("Full train csv dataset has ", train_data.shape)
# print(col_names)


# Load train_data and test_data from two separate csv files.
df = pd.read_csv('C:\\Users\\Alphacat\\Desktop\\CA ICU Project\\test.csv')#, header=None)
ycol=df[['hospital_death']]
print(ycol.shape)
del df['hospital_death']
print(df.shape)


test_onehot = pd.get_dummies(df, prefix_sep='_')
print(test_onehot.shape)
test_set = pd.concat([ycol,test_onehot],axis=1,sort=False)
test_data = test_set.values
test_col_names = list(test_set.columns.values)
print("Full test csv dataset has ", test_data.shape)


# Perform normalization on selected columns
train_data_preprocess = []
for i in range(train_data.shape[1]):
    data_temp = []
    if (i==0): # first column 'class'
        #normalize the numeric data
        catBinarizer = LabelBinarizer()
        data_temp = catBinarizer.fit_transform(train_data[:,i])
    else:
        data_temp = minmax_scale(train_data[:,i].astype(float))
        data_temp = np.reshape(data_temp, (len(data_temp),1))
    if len(train_data_preprocess) == 0:
        train_data_preprocess = data_temp
    else:
        train_data_preprocess = np.hstack([train_data_preprocess, data_temp])

print("train_data_preprocess shape:", train_data_preprocess.shape)

test_data_preprocess = []
for i in range(test_data.shape[1]):
    data_temp = []
    if (i==0): # first column 'class'
        #normalize the numeric data
        catBinarizer = LabelBinarizer()
        data_temp = catBinarizer.fit_transform(test_data[:,i])
    else:
        data_temp = minmax_scale(test_data[:,i].astype(float))
        data_temp = np.reshape(data_temp, (len(data_temp),1))
    if len(test_data_preprocess) == 0:
        test_data_preprocess = data_temp
    else:
        test_data_preprocess = np.hstack([test_data_preprocess, data_temp])

print("test_data_preprocess shape:", test_data_preprocess.shape)


# Select model input and model output
x_train = train_data_preprocess[:,1:]
y_train = train_data_preprocess[:,0]
x_test = test_data_preprocess[:,1:]
y_test = test_data_preprocess[:,0]
print("Training data:", x_train.shape, y_train.shape)
print("Test data:", x_test.shape, y_test.shape)


# define our MLP network
model = Sequential()
model.add(Dense(20, input_dim=x_train.shape[1], activation="relu",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dense(6, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dense(1, activation="sigmoid",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.summary()
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False))


# Training
class_weights = {0: 1.0,
                1: 1.0}
hist = model.fit(x_train, y_train, epochs=500,class_weight=class_weights, verbose = 1)



# Evaluate the confusion matrix using the test data
y_predict_class = model.predict_classes(x_train)
print(pd.DataFrame(confusion_matrix(y_train, y_predict_class), index=['true:0', 'true:1'], columns=['pred:0', 'pred:1']))


# Evaluate the confusion matrix using the test data
y_predict_class = model.predict_classes(x_test)
print(pd.DataFrame(confusion_matrix(y_test, y_predict_class), index=['true:0', 'true:1'], columns=['pred:0', 'pred:1']))


# predict probabilities for test set
yhat_probs = model.predict(x_test, verbose=1)
# predict crisp classes for test set
yhat_classes = model.predict_classes(x_test, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
print(yhat_probs)


# ROC AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, yhat_probs)
print('ROC AUC: %f' % auc)