from pydoc import text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import restOfLine

import Model as AI
from time import sleep

#Mengimport data testing
dftest = pd.read_csv('data_test.csv', delimiter=';')
dftest.drop('No. Urut', axis=1, inplace=True)
dftest.dropna(inplace=True) ##

#Mengimport data training
dftrain = pd.read_csv('data_train.csv', delimiter=';')
dftrain.drop('No', axis=1, inplace=True)

#Mengatasi data Null di kolom Huruf
dftrain['HURUF'].fillna(value='NA', inplace=True)

#Mengelompokkan kolom zone
cols = dftrain.columns[1:]

#Mencoba metode Z score
dfz = dftrain.copy()
for i in cols:
    df_std = np.sqrt(np.sum((dftrain[i] - dftrain[i].mean())**2) / (len(dftrain[i]) - 1))

    dfz[i] = (dftrain[i]-dftrain[i].mean()) / df_std
# print(dfz.head())

#Menggunakan metode Minmaxscaler
# dfmms = dftrain.copy()
# for a in cols:
#     dfmms[a] = (dftrain[a] - dftrain[a].min()) / (dftrain[a].max() - dftrain[a].min())
# print(dfmms.head())

#Mempersiapkan data untuk training
x = dfz.drop(['HURUF'], axis=1)
y = dfz['HURUF']

x = x.sample(frac=1)

#----------------------------------fungsi----------------------------------
def TombolPrediksi(z1,z2,z3,z4,
                   z5,z6,z7,z8,
                   z9,z10,z11,z12,
                   z13,z14,z15,z16, k):
    
    arrayX = np.array([int(z1), int(z2), int(z3), int(z4),
                      int(z5), int(z6), int(z7), int(z8),
                      int(z9), int(z10), int(z11), int(z12),
                      int(z13), int(z14), int(z15), int(z16)])
    
    x_zscore = (arrayX - np.mean(arrayX)) / np.std(arrayX)
    
    global x
    global y

    model = AI.KNN(int(k))
    model.train(x, y)
    hasil = model.predict(x_zscore)[0]
    print(hasil)
    
    global label_img
    path = f'C:/Users/SMK Telkom Malang/Documents/TEAM 2 AI/aksara/{hasil}.png'

    img = Image.open(path)
    img = img.resize((300,300))
    img.show()
    img_resized = ImageTk.PhotoImage(img)
    label_img = tk.Label(frame_right, borderwidth=1, relief='solid', text=f'{hasil}', font=('Arial', 100))
    label_img.place(width=300, height=300, relx=0.775, rely=0.35,anchor='center')





#----------------------------------gui----------------------------------
import tkinter as tk
from PIL import Image,ImageTk
from tkinter import messagebox
from tkinter import font

window = tk.Tk()
window.geometry('1200x700')
window.title('Team-2')
window.minsize(1200,700)
window.maxsize(1200,700)

#frame
frame_top = tk.Frame(window, bg='#162640').place(relwidth=1, height=50, relx=0, rely=0)
main_frame = tk.Frame(window, bg='#1570bd').place(relwidth=1, relheight=1, relx=0, y=50)
frame_bottom = tk.Frame(window, bg='#132036').place(relwidth=2, height=100,relx=0, rely=0.95, anchor='s')
layer_bottom = tk.Frame(window, bg='black').place(relwidth=2, relheight=0.05,relx=0, rely=1, anchor='s')
frame_left = tk.Frame(main_frame, bg='#1d77c1').place(relwidth=0.35, relheight=0.5, relx=0.3, rely=0.35, anchor='center')
frame_right = tk.Frame(main_frame, bg='#1d77c1').place(relwidth=0.3, relheight=0.5, relx=0.775, rely=0.35, anchor='center')

#frame_top widget
label1 = tk.Label(frame_top, text='Team-2', fg='#fdfeff', bg='#162640',font=('Arial',15, font.BOLD)).place(relx=0.1, rely=0.04, anchor='center')
label2 = tk.Label(frame_top, text='LKS Jatim 2022', fg='#fdfeff', bg='#162640',font=('Arial',15, font.BOLD)).place(relx=0.5, rely=0.04, anchor='center')
label3 = tk.Label(frame_top, text='SMKN 1 Kediri', fg='#fdfeff', bg='#162640',font=('Arial',15, font.BOLD)).place(relx=0.9, rely=0.04, anchor='center')

#frame_left widget
frame_inp = tk.Frame(frame_left, bg='#fdfeff', borderwidth=0.5, relief='solid')
frame_inp.place(width=250, height=250, relx=0.35, rely=0.35,anchor='center')

#frame_left label
zed1 = tk.Label(frame_left,fg='#fdfeff',bg='#1d77c1' ,text='Z1 - Z4', font=('Arial',12,font.BOLD)).place(relx=0.2, rely=0.22, anchor='center')
zed2 = tk.Label(frame_left,fg='#fdfeff',bg='#1d77c1', text='Z5 - Z8', font=('Arial',12,font.BOLD)).place(relx=0.2, rely=0.31, anchor='center')
zed3 = tk.Label(frame_left,fg='#fdfeff',bg='#1d77c1', text='Z9 - Z12', font=('Arial',12,font.BOLD)).place(relx=0.2, rely=0.4, anchor='center')
zed4 = tk.Label(frame_left,fg='#fdfeff',bg='#1d77c1', text='Z13 - Z16', font=('Arial',12,font.BOLD)).place(relx=0.2, rely=0.48, anchor='center')

#entry
z1 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z1.place(x=10, y=10, height=50,width=50)
z2 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z2.place(x=70, y=10, height=50,width=50)
z3 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z3.place(x=130, y=10, height=50,width=50)
z4 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z4.place(x=190, y=10, height=50,width=50)
z5 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z5.place(x=10, y=70, height=50,width=50)
z6 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z6.place(x=70, y=70, height=50,width=50)
z7 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z7.place(x=130, y=70, height=50,width=50)
z8 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z8.place(x=190, y=70, height=50,width=50)
z9 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z9.place(x=10, y=130, height=50,width=50)
z10 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z10.place(x=70, y=130, height=50,width=50)
z11 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z11.place(x=130, y=130, height=50,width=50)
z12 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z12.place(x=190, y=130, height=50,width=50)
z13 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z13.place(x=10, y=190, height=50,width=50)
z14 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z14.place(x=70, y=190, height=50,width=50)
z15 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z15.place(x=130, y=190, height=50,width=50)
z16 = tk.Entry(frame_inp,font=('Arial',12), bg='#d1d1d1', relief='flat', justify='center')
z16.place(x=190, y=190, height=50,width=50)
#frame_right widget

# path = f'C:/Users/SMK Telkom Malang/Documents/TEAM 2 AI/aksara/DHA.png'
# print(path)

# img = Image.open(path)
# img = img.resize((300,300))
# img_resized = ImageTk.PhotoImage(img)
label_img = tk.Label(frame_right, borderwidth=1, relief='solid', image=None)
label_img.place(width=300, height=300, relx=0.775, rely=0.35,anchor='center')

#label_img = tk.Label(frame_right, borderwidth=1, relief='solid').place(width=300, height=300, relx=0.775, rely=0.35,anchor='center')

#K widget
k_label = tk.Label(main_frame, text='K =', fg='#fdfeff', bg='#1570bd',font=('Arial',15))
k_label.place(rely=0.65,relx=0.22, anchor='center')
k_entry = tk.Entry(main_frame, relief='flat', text='3', font=('Arial',12))
k_entry.place(relx=0.238, rely=0.65,anchor='w')

#predbtn
pred_btn = tk.Button(main_frame, borderwidth=1 ,  width=5, relief='flat', text='Predict', font=('Arial',15, font.BOLD), padx=20)
pred_btn['command'] = lambda: TombolPrediksi(z1.get(), z2.get(), z3.get(), z4.get(),z5.get(), z6.get(), z7.get(), z8.get(), z9.get(), z10.get(), z11.get(), z12.get(), z13.get(), z14.get(), z15.get(), z16.get(), k_entry.get())
pred_btn.place(relx=0.775, rely=0.65, anchor='center')

window.mainloop()