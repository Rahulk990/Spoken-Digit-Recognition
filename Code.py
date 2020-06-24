import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Importing Data
path = "DataSet\\"
files = librosa.util.find_files(path, ext=['wav']) 
files = np.asarray(files)

# Converting to Values
data = []
y = []
for file in files: 
    ld, _ = librosa.load(file, sr = 16000, mono = True)   
    data.append(ld)
    y.append(file[62])

# Extracting Features
x = []
for i in range(len(data)):
    x.append(abs(librosa.stft(data[i]).mean(axis = 1).T))
x = np.array(x)
x = x.reshape(x.shape[0], x.shape[1], 1)

# Encoding Categorical Features
y = np.array(y)
y = y.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(y).toarray()

# Splitting Dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)

# Building CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout

classifier = Sequential()
classifier.add(Convolution1D(filters = 64, kernel_size = 6, activation = 'relu', input_shape = (1025,1)))
classifier.add(Convolution1D(filters = 64, kernel_size = 6, activation = 'relu'))
classifier.add(MaxPooling1D(pool_size = 4))
classifier.add(Dropout(0.2))
classifier.add(Flatten())
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.summary()

# Fitting and Predicting
classifier.fit(x_train, y_train, epochs=20, batch_size=10)
y_pred = classifier.predict(x_test)
_, accuracy = classifier.evaluate(x_test, y_test, batch_size=16)
y_pred = (y_pred > 0.5)
y_test = (y_test == 1)

def cvt(y):
    ny = []
    for i in y:
        f = 0
        for j in range(10):
            if(i[j] == 1):
                ny.append(j)
                f = 1
        if f == 0:
            ny.append(-1)
    return ny
y_pred = cvt(y_pred)
y_test = cvt(y_test)

# Confusion Matrix
from mlxtend.evaluate import confusion_matrix
import mlxtend.plotting.plot_confusion_matrix
cm = confusion_matrix(y_test, y_pred, False, True)
fig, ax = mlxtend.plotting.plot_confusion_matrix(conf_mat=cm)
plt.show()

# Saving model
classifier.save('model.h5')
