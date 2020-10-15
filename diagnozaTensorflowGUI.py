from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tkinter as tk
import cv2
from PIL import Image, ImageTk

from tensorflow import keras
import numpy as np

image1 = None
IMG_DIM = (300, 300)

def read_image(file):
    img = cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), dsize=IMG_DIM)
    img = np.array(img)
    img = img.reshape(1, IMG_DIM[0], IMG_DIM[1],1)
    print("Image loaded and resized")
    return img

def chooseImage():
        global image1
        image1 = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpeg"),("all files","*.*")))
        img1 = Image.open(image1)
        img1 = img1.resize((550,550), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(img1)
        img = Label(window, image=render)
        img.image = render
        img.place(relx=0.2, rely=0.2)

lbl = None
def returnResult():
        global image1
        global lbl
        if image1 is not None:
                img1 = read_image(image1)
                img1_scaled = img1.astype('float32')
                img1_scaled /= 255
                img1_scaled.reshape(300,300,1,1)
                prediction = model.predict(img1_scaled)
                if prediction[0][0] < 0.5:
                        lbl['text'] = str(prediction[0][0])+'\n Nie wykryto zapalenia płuc'
                else:
                        lbl['text'] = str(prediction[0][0])+'\n Wykryto zapalenie płuc'
                lbl.place(relx=0.13, rely=0.81)
        else:
                lbl['text'] = 'Wczytaj obraz'
                lbl.place(relx=0.35, rely=0.85)


model = keras.models.load_model('Konwolucje.h5')

window = Tk()
window.title("Lung Pneumonia Detector")
window.geometry('1000x900')

lbl = Label(window, text=' ', font=("Arial Bold", 50))

btn = Button(window, text="Wczytaj zdjęcie", command=chooseImage)
btn.place(relx=0.01, rely=0.05,heigh=100, width=100)
var = "TEST"
btn2 = Button(window, text="Zbadaj", command=returnResult)
btn2.place(relx=0.4, rely=0.05,heigh=100, width=100)
window.mainloop()
