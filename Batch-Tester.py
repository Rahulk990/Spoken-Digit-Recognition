import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Importing Data
path = "Custom\\"
files = librosa.util.find_files(path, ext=['wav']) 
files = np.asarray(files)

# Converting to Values
data = []
x = []
y = []

for file in files: 
    ld, _ = librosa.load(file, sr = 16000, mono = True)   
    data.append(ld)
    y.append(file[67])

# Extracting Features
for i in range(len(data)):
    x.append(abs(librosa.stft(data[i]).mean(axis = 1).T))
x = np.array(x)
x = x.reshape(x.shape[0], x.shape[1], 1)

# Encoding Categorical Features
from sklearn.preprocessing import OneHotEncoder
y = np.array(y)
y = y.reshape(-1,1)
enc = OneHotEncoder()
y = enc.fit_transform(y).toarray()

# Loading Model
from tensorflow.keras.models import load_model
model = load_model('model.h5')
_, accuracy = model.evaluate(x, y, batch_size=1)

# Predicting
y_pred = model.predict(x)
y_pred = (y_pred > 0.5)
y = (y == 1)

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
y = cvt(y)

# Confusion Matrix
from mlxtend.evaluate import confusion_matrix
import mlxtend.plotting.plot_confusion_matrix
cm = confusion_matrix(y, y_pred, False, True)
fig, ax = mlxtend.plotting.plot_confusion_matrix(conf_mat=cm)
plt.show()

