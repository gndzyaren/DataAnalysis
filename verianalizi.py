"""
Created on Wed July 15.07.2020 12.38 2020
@author: Yaren Gündüz
"""
#1.Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#kodlar
#2.veri önişleme
#2.1.veri yükleme

veriler = pd.read_csv(r'C:\Users\yaren\Desktop\DevrimUnayMakaleleri\MakineÖğrenmesi\Bölüm1\veriler.csv')

boy = veriler[["boy"]]

boykilo = veriler[["boy","kilo"]]


#eksik veriler
eksikveriler = pd.read_csv(r'C:\Users\yaren\Desktop\DevrimUnayMakaleleri\MakineÖğrenmesi\Bölüm2\eksikveriler.csv')
#sci - kit learn
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy = "mean")
Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

#encoder Kategorik->Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
#test
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

#Numpy dizileri dataframe'e dönüşümü
sonuc = pd.DataFrame(data = ulke,index = range(22) , columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas,index = range(22) ,columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet,index = range(22),columns=['cinsiyet'])
print(sonuc3)


#Data frame birleştirme işlemi
s =pd.concat([sonuc,sonuc2]) #2 sonuc birlestiriliyor
print(s)  #Mantıklı değil bütün kolonlar full join oluyor.

t = pd.concat([sonuc,sonuc2],axis=1)
print(t) #Mantıklı bir join işlemi oldu.NaN değerleri olmamalı

#verilerin egiti ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(t, sonuc3,test_size=0.33)
#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)