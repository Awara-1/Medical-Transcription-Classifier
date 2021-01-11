#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Using NLP to classify what the correct medical speciality is for the patient's condition, based on their recorded medical transcriptions


# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from zipfile import ZipFile

import os


# In[2]:


# Extract zip folder containing medical transcriptions


# In[3]:


# Update file/ directory names accordinly to fit local machine
ZIP_NAME = 'mtsamples.csv.zip'
TARGET_DIR = 'Users/medical-transcriptions'

# Unzip into target directory
zf = ZipFile(ZIP_NAME, 'r')
print(zf)
zf.extractall(TARGET_DIR)
zf.close()

# Read filename in directory
MT_filename = os.listdir(TARGET_DIR)
print(MT_filename[1])


# In[4]:


# Read data into pandas dataframe for processing


# In[5]:


df = pd.read_csv(TARGET_DIR + '/' + MT_filename[1])
print(df.head(1))


# In[6]:


# Observe select collumns


# In[7]:


df_select = df[['transcription','medical_specialty']]


# In[8]:


print(df_select.head(1))


# In[9]:


feature = df['transcription']
print(feature)


# In[10]:


labels = df['medical_specialty']
print(labels.unique())
print('Number of unique categories are: ' + str(len(labels.unique())) )


# In[11]:


# Pre-processing to: remove punctuation, make lowercase, remove stop words, lemmatize words


# In[12]:


# Load the NLKT utilities
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# In[13]:


from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

import re
import string

x = []
x = string.punctuation
# Punctuation words to remove
print(x)


# In[14]:


stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

# Medical stopwords pulled from https://cs.stanford.edu/people/sonal/gupta14jamia_supl.pdf
med_sw = ['disease', 'diseases', 'disorder', 'symptom', 'symptoms', 'drug', 'drugs', 'problems', 'problem' , 'prob', 'probs', 'med', 'meds',
'pill', 'pills', 'medicine', 'medicines', 'medication', 'medications', 'treatment', 'treatments', 'caps', 'capsules', 'capsule',
'tablet', 'tablets', 'tabs', 'doctor', 'dr', 'dr.', 'doc', 'physician', 'physicians', 'test', 'tests', 'testing', 'specialist', 'specialists',
'side-effect', 'side-effects', 'pharmaceutical', 'pharmaceuticals', 'pharma', 'diagnosis', 'diagnose', 'diagnosed', 'exam',
'challenge', 'device', 'condition', 'conditions', 'suffer', 'suffering' , 'suffered', 'feel', 'feeling', 'prescription', 'prescribe',
'prescribed', 'over-the-counter', 'otc']

# Return word tokens for words not in stop words and punctuation characters
def black_txt(token):
    return token not in stop_words_ and token not in list(string.punctuation) and len(token)>2 and token not in med_sw

# Function to clean text
def clean_txt(text):
    clean_text = []
    clean_text2 = []
    text = re.sub("'", "",text)
    text = re.sub("(\\d|\\W)+"," ",text)    
    clean_text = [wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
    clean_text2 = [word for word in clean_text if black_txt(word)]
    return " ".join(clean_text2)


# In[15]:


# Test function to observe cleaning process


# In[16]:


df.transcription[0]


# In[17]:


clean_txt(df.transcription[0])


# In[18]:


# Create three new variables to improve the quality of the classifier:
# Polarity - check sentiment of the text [-1, 1]. -1 defines negative sentiment and 1 defines positive sentiment
# Subjectivity - check if text is objective or subjective [0,1]. 0 means objective, 1 means subjective
# Number of word in text


# In[19]:


# NOTE: Installing textblob from spyder console enabelled it on Jupyter Lab
from textblob import TextBlob, Word, Blobber


# In[20]:


def polarity_txt(text):
    try:
        return TextBlob(text).sentiment[0]
    except:
        return None # For non-string cells to ensure comformity with Textblob


# In[21]:


def subj_txt(text):
    try:
        return TextBlob(text).sentiment[1]
    except:
        return None # For non-string cells to ensure comformity with Textblob


# In[22]:


def len_txt(text):
    try:
        if len(text.split()) > 0:
            return len(set(clean_txt(text).split()))/ len(text.split()) # Use set to return only distinct elements
        else:
            return 0
    except:
        return None # For non-values that do not confirm to split() function


# In[23]:


len_txt(df.transcription[0])


# In[24]:


polarity_txt(df.transcription[0])


# In[25]:


# Now add the new collumns in by applying the fucntions


# In[26]:


df['polarity'] = df['transcription'].apply(polarity_txt)


# In[27]:


df['subjectivity'] = df['transcription'].apply(subj_txt)


# In[28]:


df['clean_length/length'] = df['transcription'].apply(len_txt)


# In[29]:


print(df.head(5))


# In[30]:


# Transform word sequences into numerical features: vectorization - using the Term Frequency-Inverse Document 
# Frequency (TF-IDF) technique. TF-IDFT gives a higher weight to rare or less frequent words and a lower weight for
# more frequent terms


# In[31]:


# For our LSTM Deep Learning (RNN) model, tokenizer methods will be from the TF-IDFT vectorizer


# In[32]:


# For embeddings we will use spacy vector embedding (for giving words multi-dimensional meaning representations)


# In[33]:


# NOTE: Installing spacy from spyder console enabelled it on Jupyter Lab
import spacy


# In[34]:


# Error reading as filepath not pointing correctly. See solution below
# nlp = spacy.load('en_core_web_sm') 


# In[35]:


# Find spacy package path and paste path directory into load
nlp = spacy.load(r'C:\Users\ozzya\anaconda3\envs\coursera\Lib\site-packages\en_core_web_sm\en_core_web_sm-2.3.1')


# In[140]:


# Import all required tensorflow & keras libraries
import tensorflow as tf

# NOTE: Installing keras from spyder console enabelled it on Jupyter Lab
import keras

from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, Embedding
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import np_utils
from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from keras.layers import Reshape, merge, Concatenate, Lambda, Average
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


# In[141]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import time

# Feature 1
X = df['transcription']

# Labels
y = df['medical_specialty']

# Encode categories with labels of value between 0 and n_categories-1
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# One-hot encode each category based on value between 0 and n_categories-1
Y = np_utils.to_categorical(y)

# Create teh TF-IDF vector
    # max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
    # max_df = 25 means "ignore terms that appear in more than 25 documents".
    # min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
    # min_df = 5 means "ignore terms that appear in less than 5 documents".
vectorizer = TfidfVectorizer(min_df=3 , max_df=0.2, max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = None, preprocessor=clean_txt)

# Display vectorizer in action on text corpus
# Display size
SAMPLE = 20

# Fit text onto vectorizer
tfidf_wm = vectorizer.fit_transform(X[:SAMPLE])

# Create df index text
index = []
for i in range(0, SAMPLE, 1):
    index.append('Text_Row_'+str(i))

# Get vectorized words as tokens and display
tfidf_tokens = vectorizer.get_feature_names()
print(tfidf_tokens)
print(len(tfidf_tokens))

# Display text-term matrix
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens, index = [index])
print(df_tfidfvect)


# In[142]:


# Split data into train and test tests
seed = 40
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed, stratify =y)
# Fit training set onto vectorizer
vectorizer.fit(x_train.values.astype('U')) #Ensure unitype for NaN objects that can't be parsed through

print(x_train.shape)
print(y_train.shape)


# In[143]:


# Tokenize words into sequences for train and test sets


# In[144]:


word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
tokenize = vectorizer.build_tokenizer()
preprocess = vectorizer.build_preprocessor()

def to_sequence(tokenizer, preprocessor, index, text):
    words = tokenizer(preprocessor(text))
    indexes = [index[word] for word in words if word in index]
    return indexes

X_train_sequences = [to_sequence(tokenize, preprocess, word2idx, str(x)) for x in x_train]
print(x_train[10])
print(X_train_sequences[10])


# In[41]:


# THE CODE BELOW IS NOT PART OF THE MAIN BODY - IT IS TO COMPARE STANDARD TOKENIZATION WITH 
# VECTORIZED TF-IDF TOKENIZATION ABOVE

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define parameters
vocab_size = len(tfidf_tokens)
max_length = 400
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size , oov_token=oov_tok)
tokenizer.fit_on_texts(x_train.values.astype('str'))
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(x_train.values.astype('str'))

train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

# Print a sentence
print(x_train[10])

# Print tokenizsed form
print(train_sequences[10])
print(len(train_sequences[10]))

# Print tokenized + padded form
print(train_padded[10])
print(len(train_padded[10]))


# In[145]:


# Determine the max word length of cleaned text
MAX_SEQ_LENGTH = df['transcription'].astype('str').apply(clean_txt).str.split().str.len().max()

print("The maximum length in words is : " +  str(MAX_SEQ_LENGTH))


# In[146]:


N_FEATURES = len(vectorizer.get_feature_names())
print("The number of features in vectorizer is: " +  str(N_FEATURES)) 

# Pad sequences
X_train_sequences = pad_sequences(X_train_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)
print(X_train_sequences[0])

# Length of padded should be = MAX_SEQ_LENGHT
print(len(X_train_sequences[0]))


# In[147]:


# Apply to test sequences
X_test_sequences = [to_sequence(tokenize, preprocess, word2idx, str(x)) for x in x_test]
X_test_sequences = pad_sequences(X_test_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)


# In[148]:


# Make the spacy embeddings - (vectorized word embeddings to incorporate into our LSTM model)


# In[149]:


EMBEDDINGS_LEN = 300

embeddings_index = np.zeros((len(vectorizer.get_feature_names()) + 1, EMBEDDINGS_LEN))
for word, idx in word2idx.items():
    try:
        embedding = nlp.vocab[word].vector
        embeddings_index[idx] = embedding
    except:
        pass
      
print("EMBEDDINGS_LEN=", EMBEDDINGS_LEN)


# In[150]:


# LSTM Model


# In[151]:


df.info()


# In[152]:


MAX_SEQ_LENGTH = 1533

model = Sequential()
model.add(Embedding(len(vectorizer.get_feature_names()) + 1, 
                    EMBEDDINGS_LEN, 
                    weights=[embeddings_index], 
                    input_length=MAX_SEQ_LENGTH,
                    trainable=False))
model.add((LSTM(300, dropout=0.1)))
model.add(Dense(len(set(y)), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[153]:


history = model.fit(X_train_sequences, y_train, epochs=15, batch_size=64, verbose=1, validation_split=0.1)

scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1])


# In[154]:


import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


# In[ ]:




