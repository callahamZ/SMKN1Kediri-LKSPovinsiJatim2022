import numpy as np
import pandas as pd
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.K = k

    def train(self, xtrain, ytrain):

        if isinstance(xtrain, pd.DataFrame):
            xtrain = xtrain.values
        
        
        self.x_train = xtrain
        self.y_train = ytrain
    
    def predict(self, xpred):
        if isinstance(xpred, pd.DataFrame):
            xpred = xpred.values
        #Melakukan prediksi untuk setiap nilai x yang dimasukkan ke dalam xpred
        y_pred = [self._prediksi(x) for x in xpred]
        return np.array(y_pred)

    def _prediksi(self, x):
        #1. Menghitung jarak ke semua titik
        jarak_titik = [self.hitungjarak(x, x_piece) for x_piece in self.x_train]

        #2. Mengambil Titik terdekat sebanyak "K"
        k_terbaik = np.argsort(jarak_titik)[:self.K]

        #3. Mengambil label class dari k_terbaik
        label_k_terbaik = [self.y_train[i] for i in k_terbaik]

        #4. Melakukan voting label mayoritas
        hasil_voting = Counter(label_k_terbaik).most_common(1)
        return hasil_voting[0][0]

    def hitungjarak(self, x1, x2):

        #Euclidean distance
        jarak = np.sqrt(np.sum((x1 - x2)**2))
        return jarak