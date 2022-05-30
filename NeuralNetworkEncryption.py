from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np

import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.layers import Bidirectional
from keras.models import model_from_json
import pickle

main = tkinter.Tk()
main.title("Encryption And Decryption Algorithm Based On Neural Network") 
main.geometry("1300x1200")

global filename
global classifier
global char_to_int
global int_to_char
vocab_list = []
dataX = []
dataY = []
global n_vocab
global encrypt

def getID(chars,data):
    index = 0
    for i in range(len(chars)):
        if chars[i] == data:
            index = i;
            break
    return index       

def generateKey():
    global n_vocab
    dataX.clear()
    dataY.clear()
    global char_to_int
    global int_to_char
    global filename
    text.delete('1.0', END)
    sentences = ''
    with open('model/input.txt', "r") as file:
        for line in file:
            line = line.strip('\n')
            line = line.strip()
            line.lower()
            sentences+=line
    file.close()
    sentences = sentences.strip()
    vocab_list.clear()
    for i in range(len(sentences)):
        vocab_list.append(sentences[i])
    raw_text = sentences
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    n_chars = len(raw_text)
    n_vocab = len(chars)
    text.insert(END,"Key Generation Task Completed\n")
    for i in range(0, n_chars):    
        dataX.append(char_to_int.get(raw_text[i]))
        dataY.append(getID(chars,raw_text[i]))       
    text.insert(END,"Generated Key : "+str(char_to_int['w'])+str(char_to_int['p'])+str(char_to_int['l'])+str(char_to_int['e'])+str(char_to_int['A'])+str(char_to_int['b'])+"\n")

def buildModel():
    global classifier
    text.delete('1.0', END)
    n_patterns = len(dataX)
    if os.path.exists('model/nn_model.json'):
        with open('model/nn_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()
        classifier.load_weights("model/nn_model_weights.h5")
        classifier._make_predict_function()
    else:
        seq_length = 1
        X = np.reshape(dataX, (n_patterns, seq_length, 1))
        X = X / float(n_vocab)
        y = np_utils.to_categorical(dataY)
        print(X.shape)
        print(y.shape)
        model = Sequential()
        model.add(Bidirectional(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(256)))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        hist = model.fit(X, y, epochs=8000, batch_size=64)
        model.save_weights('model/nn_model_weights.h5')            
        model_json = model.to_json()
        with open("model/nn_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/nn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    f = open('model/nn_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    loss = data['loss']
    lossValue = loss[7999]
    loss = loss[0:100]
    text.insert(END,"Neural Network Training Model Loss = "+str(lossValue)+"\n")
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch/Iterations')
    plt.ylabel('Loss')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['Neural Network Loss'], loc='upper left')
    plt.title('Neural Network Loss Graph')
    plt.show()

def decimalToBinary(n):
    return "{0:b}".format(int(n))

def encryption():
    text.delete('1.0', END)
    global encrypt
    global classifier
    encrypt = []
    message = tf1.get();
    binValue = ''
    for i in range(len(message)):
        data = char_to_int[message[i]]
        temp = []
        temp.append(data)
        temp = np.asarray(temp)
        x = np.reshape(temp, (1, temp.shape[0], 1))
        x = x / float(n_vocab)
        encrypted = classifier.predict(x, verbose=0)[0]
        encrypt.append(np.argmax(encrypted))
        binValue+=str(decimalToBinary(np.argmax(encrypted)))+" "
    text.insert(END,"Original Message : "+message+'\n\n')
    text.insert(END,"Encrypted Message Matrix : "+str(encrypt)+"\n\n")
    text.insert(END,"Encrypted Binary Value   : "+str(binValue.strip())+"\n\n")

def decryption():
    text.delete('1.0', END)
    global encrypt
    global classifier
    encrypt = np.asarray(encrypt)
    output = ''
    for i in range(len(encrypt)):
        index = encrypt[i]
        result = int_to_char[index]
        output+=result
    text.insert(END,"Decrypted Message : "+str(output)+"\n\n")    
    
def close():
    main.destroy()

    
font = ('times', 16, 'bold')
title = Label(main, text='Encryption And Decryption Algorithm Based On Neural Network')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=17,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=170)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Generate Key", command=generateKey, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)

lstmButton1 = Button(main, text="Build Neural Network Model", command=buildModel, bg='#ffb3fe')
lstmButton1.place(x=350,y=550)
lstmButton1.config(font=font1) 

l1 = Label(main, text='Enter Message')
l1.config(font=font1)
l1.place(x=50,y=100)

tf1 = Entry(main,width=40)
tf1.config(font=font1)
tf1.place(x=230,y=100)

gruButton = Button(main, text="Neural Network Data Encryption", command=encryption, bg='#ffb3fe')
gruButton.place(x=50,y=600)
gruButton.config(font=font1) 

graphButton = Button(main, text="Neural Network Data Decryption", command=decryption, bg='#ffb3fe')
graphButton.place(x=350,y=600)
graphButton.config(font=font1) 

predictButton = Button(main, text="Exit", command=close, bg='#ffb3fe')
predictButton.place(x=630,y=600)
predictButton.config(font=font1) 

main.config(bg='LightSalmon3')
main.mainloop()
