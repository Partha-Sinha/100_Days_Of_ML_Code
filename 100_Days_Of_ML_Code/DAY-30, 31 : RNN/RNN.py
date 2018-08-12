# Recurrent Neural Network

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Training set
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the outputs
X_train = training_set[0:1257]
y_train = training_set[1:1258]

# Reshaping
X_train = np.reshape(X_train, (1257, 1, 1))

# Importing the Keras Libraries and Packages
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output Layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)
