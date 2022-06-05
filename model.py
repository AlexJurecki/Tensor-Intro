import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Adam

df=pd.read_csv("admissions_data.csv")
print (df.describe())
print(np.shape(df))
df.drop(columns = ["Serial No."])
labels = df.iloc[:,-1]
features = df.iloc[:,:-1]


features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.3,random_state = 1)

numerical_features=features.select_dtypes(
  include = ['float64','int64']
)
numerical_columns=numerical_features.columns

ct=ColumnTransformer([("only numeric",StandardScaler(),numerical_columns)],remainder="passthrough")

features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.fit_transform(features_test)
# Do extensions code below
# if you decide to do the Matplotlib extension, you must save your plot in the directory by uncommenting the line of code below

def design_model(X,learning_rate):
  my_model=Sequential()
  my_input=layers.InputLayer(input_shape=(X.shape[1],))
  my_model.add(my_input)
  my_model.add(Dense(64,activation='relu'))
  my_model.add(Dense(1))
  opt = Adam(learning_rate=learning_rate)
  my_model.compile(
    loss='mse',
    metrics=['mae'],
    optimizer=opt)

  return my_model

def fit_model(features_train,labels_train,learning_rate,num_epochs):
  model=design_model(features_train,learning_rate)
  stop=EarlyStopping(monitor='val_loss',
  mode='min',
  verbose=1,
  patience=50)
  history=model.fit(features_train,labels_train,epochs=num_epochs,batch_size=20,verbose=0,validation_split=0.2,callbacks=[stop])
  return history

learning_rate=0.01
num_epochs=10000
history=fit_model(features_train_scaled,labels_train,learning_rate,num_epochs)
for property in vars(history).items():
  pass


val_mse, val_mae = history.model.evaluate(
  features_test, 
  labels_test, 
  verbose = 0
)

print("MAE: ", val_mae)


print(history.history.keys())
print("MSE: ", val_mse)


fig = plt.figure(figsize = (100,30))
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
 
  # Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping each other  
fig.tight_layout()


fig.savefig('static/images/my_plots.png')