# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 12:10:32 2019

@author: Monster
"""
"""

NLP kullanarak Veri kumesinde bulunan yorumlar ile liked degerleri arasindaki iliskiyi makineye ogretmeye
ve bu ogrenimi test etmeye calisicaz

NLP Adimlari:
    
    Veri kaynagi -> Data PreProcessing -> Feature Extraction  -> Machine Learning -> Results
 

Stopwords -> Herhangi bir dilde anlami olmayan kelimeler toplulugu  Exp: I, the, is, didn't, as, out, then ...

"""

import pandas as pd

datas = pd.read_csv("Restaurant_Reviews.csv")

# Preprocessing asamasi (Stopwords, Case sensitive gibi islemlerle verileri filtreden geciricez)

import re                                      # Regular Expression                                                        
import nltk
# nltk.download("stopwords")  denemek icin stopwords(ingilizce icin) veri kumesi indirilebilir 

from nltk.stem.porter import PorterStemmer     # PorterStemmer ile kelimeri koklerine ayiralim
ps = PorterStemmer()

from nltk.corpus import stopwords
ara =[]
new_reviews =[]
for i in range(1000):
    # Alfanumeric karakterleri filtrele
    reviews = re.sub('[^a-zA-Z]',' ',datas['Review'][i])    # yorumlar icin [a-zA-Z] icermeyen karakterleri 'space character' ile degistir
    # harfleri kucult
    reviews = reviews.lower()
    # bosluk karakterleri ile parcala ve bir list olustur
    reviews = reviews.split()
    
    # stopwords de bulunan butun ingilizce stopword leri set et
    # eger reviews listesinde bulunan kelimeler set edilen kumede yoksa yeni olusturulan listeye stem() fonk ile kelimenin govdesini ekle 
    # exp: ['wow', 'loved', 'this', 'place'] -> ['wow', 'love', 'place']
    reviews = [ps.stem(word) for word in reviews if not word in set (stopwords.words("english")) ]  
   
    # olusturan listeyi ' ' ile birlestirerek string olustur 
    reviews = ' '.join(reviews) 
    new_reviews.append(reviews)


# Feature Extraction asamasindayiz..

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer( max_features = 1000 )  # max_features -> en cok kullanilan kelime sayisi

# X -> bagimsiz , Y-> bagimli degisken 
X = cv.fit_transform(new_reviews).toarray() # cumlelerde yer alan kelimeler her satir icin sayisal degerlere gruplandi
Y = datas.iloc[:,1].values

# Machine Learning asamasina gecebiliriz 








