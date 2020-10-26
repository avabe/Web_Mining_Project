import json
import os
import nltk
import re
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import pylab as pl
import tweepy
from tweepy import OAuthHandler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.layers import Dropout
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from collections import Counter

path = os.getcwd()
dataset = pd.read_csv(path+ '/training.csv', encoding='ISO-8859-1', header=None).sample(60000)
print("Our data set looks like that:")
dataset.head()


# --------------- Question 1 - text pre-processing ---------------


def preprocess(tweet):
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)

    # Convert @username to __USERHANDLE
    tweet = re.sub('@[^\s]+', '__USERHANDLE', tweet)

    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    # trim
    tweet = tweet.strip('\'"')

    # Repeating words like hellloooo
    repeat_char = re.compile(r"(.)\1{1,}", re.IGNORECASE)
    tweet = repeat_char.sub(r"\1\1", tweet)

    # Emoticons
    emoticons = \
        [
            ('__positive__', [':-)', ':)', '(:', '(-:', \
                              ':-D', ':D', 'X-D', 'XD', 'xD', \
                              '<3', ':\*', ';-)', ';)', ';-D', ';D', '(;', '(-;', ]), \
            ('__negative__', [':-(', ':(', '(:', '(-:', ':,(', \
                              ':\'(', ':"(', ':((', 'D:']), \
            ]

    def replace_parenthesis(arr):
        return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

    def join_parenthesis(arr):
        return '(' + '|'.join(arr) + ')'

    emoticons_regex = [(repl, re.compile(join_parenthesis(replace_parenthesis(regx)))) \
                       for (repl, regx) in emoticons]

    for (repl, regx) in emoticons_regex:
        tweet = re.sub(regx, ' ' + repl + ' ', tweet)

    # Convert to lower case
    tweet = tweet.lower()

    # Stop Words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    tweet = ' '.join(filtered_sentence)

    return tweet


# Stemming of Tweets

def stem(tweet):
    stemmer = nltk.stem.PorterStemmer()
    words = [word if (word[0:2] == '__') else word.lower()
             for word in tweet.split()
             if len(word) >= 3]
    words = [stemmer.stem(w) for w in words]
    tweet_stem = ' '.join(words)
    return tweet_stem


# filtering the sentiments from the data
# positive sentiments
df1 = dataset[dataset[0] == 4].iloc[:,5].values
possitive_sentiment = [stem(preprocess(tweet)) for tweet in df1]

vec1 = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True,use_idf = True,ngram_range=(1, 2))
# possitive vec
possitive_sentiment_vec = vec1.fit_transform(possitive_sentiment)
feature_names = vec1.get_feature_names()

df = pd.DataFrame(possitive_sentiment_vec.T.todense(), index=feature_names)
# Export data to csv
# open('train_positive_freq.csv').close()
# df.to_csv(path + '\\train_positive_freq.csv')
print("tf-idf scores in a matrix for possitive sentiment:")
print(df)

# negative sentiments
df2 = dataset[dataset[0] == 0].iloc[:,5].values
negative_sentiment = [stem(preprocess(tweet)) for tweet in df2]
# negative vec
vec2 = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True,use_idf = True,ngram_range=(1, 2))
negative_sentiment_vec = vec2.fit_transform(negative_sentiment)
feature_names = vec2.get_feature_names()

df = pd.DataFrame(negative_sentiment_vec.T.todense(), index=feature_names)
# Export data to csv
# df.to_csv(path + '\\train_negative_freq.csv')
print("tf-idf scores in a matrix for negative sentiment:")
print(df)


dataset[0].value_counts().sort_index().plot.bar()

print("Text length:")
dataset[5].str.len().plot.hist()


X = dataset.iloc[:, 5].values
X = pd.Series(X)
y = dataset.iloc[:, 0].values
print(y)
X = [stem(preprocess(tweet)) for tweet in X]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)


# --------------- Question 2 - Training a machine learning model ---------------

# ------- Model 1 : Multinomial Naive Bayes classifier using 2gram --------

# tfidf vector
vec = TfidfVectorizer()
vec.fit(X_train)
X_train_vec = vec.transform(X_train)
nb = MultinomialNB()
nb.fit(X_train_vec,y_train)
X_test_vec = vec.transform(X_test)
pred = nb.predict(X_test_vec)

print(metrics.accuracy_score(y_test, pred))
print("mean: "+str(y_test.mean()))
print("1-mean: "+str(1-y_test.mean()))


nb = MultinomialNB(alpha=0.5)
nb.fit(X_train_vec,y_train)
pred2 = nb.predict(X_test_vec)

print(metrics.accuracy_score(y_test, pred2))


nb = MultinomialNB(alpha=0.8)
nb.fit(X_train_vec,y_train)
pred3 = nb.predict(X_test_vec)

print(metrics.accuracy_score(y_test, pred3))


print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, pred))
# [[128420  31369]
# [ 33164 127047]]
cm = metrics.confusion_matrix(y_test, pred)
pl.matshow(cm)
# pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()

print(metrics.classification_report(y_test, pred))


# -------- Model 2 : Recurrent Neural Network using 2gram to vec -------

tokenizer = Tokenizer(num_words=5000, split=" ")
tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X) # padding our text vector so they all have the same length
y = pd.get_dummies(y).values
print("The data vector:\n")
print(X[:5])


batch_size = 32
epochs = 3
model = Sequential()
model.add(Embedding(5000, 256, input_length=X.shape[1]))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

history = model.fit(X_train, y_train, epochs=epochs,validation_split=0.1, batch_size=batch_size, verbose=1,callbacks=callbacks)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


def pred(score):
    p=[0,0]
    if score[0] < 0.5:
        p[0]=0
    else:
        p[0]=1
    if score[1] < 0.5:
            p[1]=0
    else:
            p[1]=1
    return p

def valp(p):
    if p[0]==1 and p[1]==0:
        return 0
    else:
        return 4

y_pred_1d = []
y_test_1d = list(y_test)
scores = model.predict(X_test, verbose=1, batch_size=8000)
y_pred_1d = [pred(score) for score in scores]


y_test_v = [valp(p) for p in y_test]
y_pred_v = [valp(p) for p in y_pred_1d]
cm = metrics.confusion_matrix(y_test_v, y_pred_v)
print("Confusion matrix:")
print(cm)
print("mean: "+str(y_test.mean()))
print("1-mean: "+str(1-y_test.mean()))
pl.matshow(cm)
# pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()
print(metrics.classification_report(y_test_v, y_pred_v))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


history = model.fit(X_train, y_train, epochs=5,validation_split=0.1, batch_size=batch_size, verbose=1,callbacks=callbacks)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))


y_pred_1d = []
y_test_1d = list(y_test)
scores = model.predict(X_test, verbose=1, batch_size=8000)
y_pred_1d = [pred(score) for score in scores]


y_test_v = [valp(p) for p in y_test]
y_pred_v = [valp(p) for p in y_pred_1d]
cm = metrics.confusion_matrix(y_test_v, y_pred_v)
print("Confusion matrix:")
print(cm)
print("mean: "+str(y_test.mean()))
print("1-mean: "+str(1-y_test.mean()))
pl.matshow(cm)
#pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()
print(metrics.classification_report(y_test_v, y_pred_v))


# -------------- Question 3 ----------------

consumer_key = 'DX8snsfc5tS1HZLwh3Smv5kIm'
consumer_secret = 'jFgJa36aCy81BwDoVQaV2KXORILBqZytbj18f7F68yctBhAZ5v'
access_token = '915695525747359744-YhdmKVnD85PmOZFhCrs9Je3WZvORq58'
access_secret = '3wfubaVXRnI3uEfl3k7ZSUgz2bPrkJDe0VSlXuhAzqEmu'

auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

# Global variables to retrain the number of tweets we collect
tweet_count = 0
# Input number of tweets to be downloaded
n_tweets = 7500

# Erasing the files contents
open('positive.json', 'w').close
open('negative.json', 'w').close


# override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):

    def __init__(self, file_name):
        self.file_name = file_name

    def on_data(self, data):
        global tweet_count
        global n_tweets
        global myStream
        if tweet_count < n_tweets:
            try:
                with open(self.file_name, 'a') as f:
                    f.write(data)
                    tweet_count += 1
                    return True
            except BaseException as e:
                print("Error on_data: %s" % str(e))
            return True
        else:
            myStream.disconnect()

    def on_error(self, status):
        print(status)
        return True

    def on_exception(self, exception):
        print(exception)
        return


myStreamListener = MyStreamListener('positive.json')

while tweet_count < n_tweets:
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
    try:
        myStream.filter(track=['good', 'nice', 'great', 'awesome', 'outstanding','impressive','fantastic',
                       'terrific', 'like', 'love'])
    except Exception as e:
        continue

print('done!')

myStreamListener = MyStreamListener('negative.json')

tweet_count = 0

while tweet_count < n_tweets:
    myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
    try:
        myStream.filter(track=['bad', 'terrible', 'crap', 'useless', 'hate', 'disappointing', 'sad', 'mad'])
    except Exception as e:
        continue

print('done!')

positive_data = []
negative_data = []

with open('positive.json', 'r') as f:
    for line in f:
        if len(line) <= 0 or line == '\n':
            continue
        positive_data.append(json.loads(line))

with open('negative.json', 'r') as f:
    for line in f:
        if len(line) <= 0 or line == '\n':
            continue
        negative_data.append(json.loads(line))

positive_tweets = []
negative_tweets = []

for tweet in positive_data:
    if 'text' in tweet:
        positive_tweets.append(stem(preprocess(tweet['text'])))

for tweet in negative_data:
    if 'text' in tweet:
        negative_tweets.append(stem(preprocess(tweet['text'])))

print('Positive tweets:')
print(positive_tweets[:5])
print('Negative tweets:')
print(negative_tweets[:5])


count_positive = Counter()
for tweet in positive_tweets:
    terms = tweet.split()
    if 'url' in terms:
        terms.remove('url')
    count_positive.update(terms)

count_negative = Counter()
for tweet in negative_tweets:
    terms = tweet.split()
    if 'url' in terms:
        terms.remove('url')
    count_negative.update(terms)


print('Most popular terms for positive sentiment:')
print(count_positive.most_common(10))
print('Most popular terms for negative sentiment:')
print(count_negative.most_common(10))


positive_vec = TfidfVectorizer()
positive_terms = positive_vec.fit_transform(positive_tweets)

feature_names = positive_vec.get_feature_names()

df = pd.DataFrame(positive_terms.T.todense(), index=feature_names)
#Export data to csv
#df.to_csv(path + '\\stream_positive_freq.csv')

print("Terms frequency for positive sentiment:")
print(df)


negative_vec = TfidfVectorizer()
negative_terms = negative_vec.fit_transform(negative_tweets)

feature_names = negative_vec.get_feature_names()

df = pd.DataFrame(negative_terms.T.todense(), index=feature_names)
#Export data to csv
#df.to_csv(path + '\\stream_negative_freq.csv')

print("Terms frequency for negative sentiment:")
print(df)


# -------------- Question 4 ----------------

X_test = vec.transform(positive_tweets + negative_tweets)
Y_test = [4] * len(positive_tweets) + [0] * len(negative_tweets)

pred = nb.predict(X_test)

print('Accuracy score:')
print(metrics.accuracy_score(Y_test, pred))


print('Confusion matrix:')
cm = metrics.confusion_matrix(Y_test, pred)

print(metrics.classification_report(Y_test, pred))












