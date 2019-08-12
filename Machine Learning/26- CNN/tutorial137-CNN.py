# -*- coding: utf-8 -*-

"""
train ve test veri kumesini http://www.bilkav.com/wp-content/uploads/2018/08/27CNN_Cinsiyet.zip icerisinden alinabilir.
Burada train ve test klasorlerinde bulunan ornek fotograflari kullanarak, 
CNN uzerinden kadin ve erkek tahminlerini ve bu tahminlerin dogrulugunu test edicez.
"""

from keras.models import Sequential      # ANN tanimlama icin
from keras.layers import Convolution2D   
from keras.layers import MaxPooling2D
from keras.layers import Flatten      
from keras.layers import Dense          # layer eklemek icin


classifier = Sequential()   # keras kullanarak bir ANN olustur
# ANN tanimlandigina gore simdi olusturalim

# 1- Convolution
# okunan resimler 64*64*3 -> height, width, color(RGB)
classifier.add(Convolution2D(32, 3, 3, input_shape= (64, 64, 3), activation='relu'))

# 2- Pooling 
classifier.add(MaxPooling2D(pool_size= (2, 2)))     # Max pooling


# 3- 2. Convolution katmani
classifier.add(Convolution2D(32, 3, 3, activation='relu'))

# 4- 2. Pooling 
classifier.add(MaxPooling2D(pool_size= (2, 2)))     

# 5- Flattening
classifier.add(Flatten())                           # Duzlestirme

# 6- ANN(Yapay sinir Agi)

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim= 1, activation = 'sigmoid'))    # output layer, tavsiye edilen 'sigmoid' function

# CNN
classifier.compile(optimizer='adam', loss= 'binary_crossentropy',metrics=['accuracy'])

# CNN ve resim yukleme
from keras.preprocessing.image import ImageDataGenerator

# resimleri bir filtreleme benzeri asamadan gecirerek oku

test_datagen = ImageDataGenerator(rescale= 1./255,       # resim olceklendirmeleri 
                                  shear_range= 0.2,      # 
                                  zoom_range= 0.2,       # 
                                  horizontal_flip= True) # 

train_datagen = ImageDataGenerator(rescale= 1./255,      # ayni islemi train_datagen icin de yap
                                   shear_range= 0.2,     
                                   zoom_range= 0.2,      
                                   horizontal_flip= True)

# test ve train setlere oku
test_set = test_datagen.flow_from_directory('test_set',       # test_set -> ayni dizinde bulunan test fotograflarinin klasoru
                                                 target_size= (64,64),
                                                 batch_size= 1,
                                                 class_mode= 'binary')

training_set = train_datagen.flow_from_directory('training_set',    # training_set -> ayni dizinde bulunan trainin_set klasoru
                                                 target_size= (64,64),
                                                 batch_size= 1,
                                                 class_mode= 'binary')  


# ANN 'u train et
classifier.fit_generator(training_set,
                         steps_per_epoch = 500,       # epoch sayisi
                         epochs= 1,                   # daha iyi egitim icin arttirilmalidir
                         validation_data= test_set,
                         validation_steps= 203 )      # test sette 203 veri var


import numpy as np
import pandas as pd

test_set.reset()
predict = classifier.predict_generator(test_set, verbose=1, steps = 203)     # test_set deki her bir veri icin predict islemi

# tahmin verilerinin 1-0 transformasyonu
predict[predict > 0.5] = 1
predict[predict <= 0.5] = 0

test_labels = []     

for i in range(0, int(203)): 
    test_labels.extend(np.array(test_set[i][1]))

print("test_labels:", test_labels)

# birebir dosya tahminleri icin dosya bilgilerini alalim
file_names = test_set.filenames
result = pd.DataFrame()             # sonuclari gozlemlemek icin dataframe olusturalim
result ['file_names'] = file_names
result ['predict'] = predict
result ['test'] = test_labels

# confusion matrix ile predict doruluguna bakalim
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, predict)
print(cm)















