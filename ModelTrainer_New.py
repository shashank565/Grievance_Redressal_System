#!/usr/bin/env python
# coding: utf-8

# # import the library

# In[1]:


import re
import ftfy
import nltk
import itertools
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from nltk import PorterStemmer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import  classification_report, confusion_matrix, accuracy_score
from keras.layers import Conv1D, Dense, Input, LSTM, Embedding, Dropout, Activation, MaxPooling1D


# # Read data from folder

# In[2]:


np.random.seed(1234) 


# In[3]:


df = pd.read_csv('input/data.csv')


# In[4]:


df.head()


# In[5]:


max_length = 250
nb_max_words = 100
embedding_dim = 30


# # Data PreProcessing

# In[6]:


df.columns


# In[7]:


review = df['COMPLAINT DESCRIPTION '].values
result = df['CRIME CATEGORY'].values


# In[8]:


data = {'COMPLAINT DESCRIPTION ':review, 'CRIME CATEGORY':result}
df = pd.DataFrame(data)


# In[9]:


df.head()


# In[10]:


df['CRIME CATEGORY'].unique()


# In[11]:


df.isna().sum()


# In[12]:


df1 = df['CRIME CATEGORY'].value_counts()


# In[13]:


df1


# In[14]:


df.dropna(axis=0,inplace=True)


# In[15]:


df1 = df['CRIME CATEGORY'].value_counts()


# In[16]:


df1


# In[17]:


df.shape


# In[18]:


review = df['COMPLAINT DESCRIPTION '].values
result = df['CRIME CATEGORY'].values


# # Each class count based on verified 

# In[19]:


df1.plot.pie(y= 'Count', figsize=(7, 7),autopct='%1.1f%%')


# # Data Preprocessing

# In[20]:


cList = pkl.load(open('input/cword_dict.pkl','rb'))


# In[21]:


print(cList)


# In[22]:


c_re = re.compile('(%s)' % '|'.join(cList.keys()))


# In[23]:


c_re


# In[24]:


def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group()]
    return c_re.sub(replace, text)


# In[25]:


def clean_review(reviews):
    cleaned_review = []
    for review in reviews:
        review = str(review)
#         if re.match("(\w+:\/\/\S+)", review) == None and len(review) > 10:
        review = ' '.join(re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", review).split())
        review = ftfy.fix_text(review)
        review = expandContractions(review)
        review = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", review).split())
        stop_words = stopwords.words('english')
        word_tokens = nltk.word_tokenize(review) 
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        review = ' '.join(filtered_sentence)
        review = PorterStemmer().stem(review)
        cleaned_review.append(review)
    return cleaned_review


# In[26]:


arr_review = [x for x in df['COMPLAINT DESCRIPTION ']]


# In[27]:


arr_review


# In[28]:


cleaned_text = clean_review(arr_review)


# In[29]:


result


# In[30]:


len(cleaned_text)


# In[31]:


len(result)


# In[32]:


data = {'Review':  cleaned_text,
        'Result': result
        }


# In[33]:


df = pd.DataFrame(data, columns=['Review','Result'])


# In[34]:


df.head()


# In[35]:


df['Result'].unique()


# In[36]:


df.Result.value_counts()


# # Tokenizer

# In[37]:


tokenizer = Tokenizer(num_words=nb_max_words)
tokenizer.fit_on_texts(cleaned_text)


# In[38]:


sequences_text = tokenizer.texts_to_sequences(cleaned_text)


# In[40]:


with open('model\\tokens.pkl', 'wb') as handle:
    pkl.dump(tokenizer, handle, protocol=pkl.HIGHEST_PROTOCOL)


# In[41]:


word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))


# # Add extra zero for padding

# In[42]:


data = pad_sequences(sequences_text, maxlen=max_length)

print('Shape of data tensor:', data.shape)


# In[43]:


data[0]


# # make labels as one hot encoding

# In[44]:


from imblearn.over_sampling import RandomOverSampler


# In[45]:


ros = RandomOverSampler(random_state=42)


# In[46]:


variablex_ros, y_ros = ros.fit_resample(data, result)


# In[47]:


# fit predictor and target 

print('Original dataset shape', result.shape)
print('Resample dataset shape', y_ros.shape)


# In[48]:


labels = np.asarray(pd.get_dummies(y_ros),dtype = np.int8)


# In[49]:


labels[0]


# In[50]:


print(variablex_ros.shape, labels.shape)


# # split data into train and test

# In[51]:


X_train, X_test, Y_train, Y_test = train_test_split(variablex_ros, labels, test_size = 0.2, random_state=42)


# In[52]:


print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# # Build Model

# In[53]:


model = Sequential()

model.add(Embedding(nb_max_words, embedding_dim, input_length=250))
model.add(Dropout(0.2))

model.add(Conv1D(250, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(500, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# model.add(Conv1D(128, 5, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(5, activation='sigmoid'))


# # Parameter tuning

# In[54]:


adam = Adam(lr=0.001,
    decay=1e-06
)


# In[55]:


model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
)


# In[56]:


model.summary()


# # Train the model

# In[57]:


hist = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=5,
    batch_size=1,
    shuffle=True
)


# In[58]:


plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[59]:


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[60]:


Y_pred = model.predict(X_test)


# In[61]:


y_predict = []
for i in range(0, len(Y_pred)):
    y_predict.append(int(np.argmax(Y_pred[i])))
len(y_predict)


# In[62]:


y_true = []
for i in range(0, len(Y_test)):
    y_true.append(int(np.argmax(Y_test[i])))
len(y_true)


# # test and evaluate the model

# In[63]:


accuracy = accuracy_score(y_true, y_predict)
print("Accuracy: %.2f%%" % (accuracy*100))
lm_acc = accuracy*100


# In[64]:


print(classification_report(y_true, y_predict))


# In[65]:


def plot_confusion_matrix(cm, classes,title='Confusion matrix'):
    plt.figure(figsize=(7,7))
    plt.imshow(cm, interpolation='nearest', cmap='binary')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes, rotation=30)
    
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


# In[66]:


cm = confusion_matrix(y_pred= y_predict, y_true=y_true)


# In[67]:


cm_plot_labels = ['Robbery', 'Murder', 'Assault', 'cyber _crime', 'Accident_case']


# In[68]:


plot_confusion_matrix(cm,cm_plot_labels)


# # Store the trained model

# In[69]:


model_structure = model.to_json()
f = Path("model/model_structure.json")
f.write_text(model_structure)


# In[70]:


model.save_weights("model/model_weights.h5")


# # end
