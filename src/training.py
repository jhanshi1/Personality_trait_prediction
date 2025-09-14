import pandas as pd
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


#importing dataset
df=pd.read_csv("data/mypersonality_final.csv",encoding="latin-1")
print(df.head())
print(df.info())
print(df.isnull().sum())
df=df.dropna(subset=["STATUS"])

#Extracting datasets
texts=df["STATUS"].astype("str")
labels=df[["cEXT","cNEU","cAGR","cCON","cOPN"]]

#cleaning dataset
stop_words=set(stopwords.words("english"))
def cleaning(text):
    text=text.lower()
    text=re.sub(r"http\S+|www\S+|https\S+","",text)  #URLS
    text = re.sub(r"@\w+|#\w+", '', text)            #Mentions,hashtags
    text=re.sub(r"[^a-z\s]","",text)                 #keep only letters
    tokens=text.split()
    tokens=[w for w in tokens if w not in stop_words]
    return ' '.join(tokens)
print(texts[0])
text_cleaned=texts.apply(cleaning)
print(text_cleaned[0])

#Tokenization
MAX_VOCAB=10000
MAX_LEN=100
tokenizer=Tokenizer(num_words=MAX_VOCAB,oov_token="<oov>")
tokenizer.fit_on_texts(text_cleaned)
sequences=tokenizer.texts_to_sequences(text_cleaned)
padded=pad_sequences(sequences,maxlen=MAX_LEN,padding="post",truncating="post")
print(sequences[0])

#Train/Text split
x_train,x_test,y_train,y_test=train_test_split(padded,labels,test_size=0.2,random_state=42)
print('Train shape:',x_train.shape,y_train.shape)
print("Text shape:",x_test.shape,y_test.shape)

#TF-IDF features for ml
tfidf=TfidfVectorizer(max_features=5000)
x_train_tfidf=tfidf.fit_transform(x_train)
x_test_tfidf=tfidf.transform(x_test)
