import pandas as pd
import chardet
import numpy as np
import twitter_scraper as ts
from nltk.tokenize import TweetTokenizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
import re
import statistics 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import SpatialDropout1D
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import accuracy_score
from keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
# Preprocessing #

# Read file
input = 'all_annotated.tsv'
data = pd.read_csv(input, sep='\t', encoding="utf-8")


# Drop unecessary attributes
data = data.drop(['Tweet ID', 'Country', 'Date', 'Automatically Generated Tweets', 'Ambiguous due to Named Entities', 'Code-Switched'], axis=1)


# Tokenize tweets into arrays
token = TweetTokenizer()
#data['Tweet'] = data.apply(lambda row: re.sub('[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', row['Tweet']), axis=1)
#data['Tweet'] = data.apply(lambda row: (row['Tweet']).lower(), axis=1)
#data['Tweet'] = data.apply(lambda row: text_to_word_sequence(row['Tweet']), axis=1)


data_arr =  np.array(data['Tweet'])

print(data_arr)
# Labels: 0 = English, 1 = Ambiguous, 2 = Not English, 3 = Code Switched
label_arr = np.array(data.drop(['Tweet'], 1).astype(int))
'''
#print(label_arr[0])
labels = []
for i in range(len(label_arr)):
	if (label_arr[i][0] == 1):
		labels.append(0)
	elif (label_arr[i][1] == 1):
		labels.append(1)
	elif (label_arr[i][2] == 1):
		labels.append(2)
	elif (label_arr[i][3] == 1):
		labels.append(3)
'''
#(unique, counts) = np.unique(data_arr.flatten(), return_counts=True)
#vocab_size = len(unique)
#labels_a = np.array(labels)
token = Tokenizer()
token.fit_on_texts(data_arr)
index = token.word_index
index_len = len(index)
new_data = token.texts_to_sequences(data_arr)
new_data = pad_sequences(new_data)
print(new_data.shape)

train, test, train_lab, test_lab = train_test_split(new_data, label_arr)	
print(train.shape, train_lab.shape)	
print(test.shape, test_lab.shape)	
#embed = Word2Vec(train, min_count=1)
print(new_data.shape[1])
model = Sequential()

model.add(Embedding(index_len*2, 100, input_length=new_data.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.save_weights('start')
epochs = 10
batch = 550
"""
run = model.fit(train, train_lab, epochs=epochs, batch_size=batch, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=2, min_delta=0.001)])

results = model.evaluate(test, test_lab)

print("loss: ", results[0], " accuracy: ", results[1])

test_tweet = ['this ツ tweet is in　ツ english']

test_token = token.texts_to_sequences(test_tweet)
test_token = pad_sequences(test_token, maxlen=31)
pred = model.predict(test_token)
label_names = ['English', 'Ambiguous', 'Not English', 'Code Switched']
print(pred, label_names[np.argmax(pred)])
"""

dec_tree = tree.DecisionTreeClassifier(min_samples_leaf=150, max_depth=10)
dec_tree = dec_tree.fit(train, train_lab)
preds = dec_tree.predict(test)

print("Decision Tree accuracy: ", accuracy_score(test_lab, preds))


kfold = KFold(10, True, 1)
d_t_scores = []
d_l_scores = []
for train1, test1 in kfold.split(new_data):
	train_k = (new_data[train1])
	train_lab_k = (label_arr[train1])
	test_k = (new_data[test1])
	test_lab_k = (label_arr[test1])

	run = model.fit(train_k, train_lab_k, epochs=epochs, batch_size=batch, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=2, min_delta=0.001)])
	results = model.evaluate(test_k, test_lab_k)
	d_l_scores.append(results[1])
	model.load_weights('start')
	dec_tree = tree.DecisionTreeClassifier(min_samples_leaf=150, max_depth=10)
	dec_tree = dec_tree.fit(train_k, train_lab_k)
	preds2 = dec_tree.predict(test_k)
	acc = accuracy_score(test_lab_k, preds2)
	d_t_scores.append(acc)

print("Decision Tree accuracy after k-fold: ", statistics.mean(d_t_scores))

print("Deep Learning accuracy after k-fold: ", statistics.mean(d_l_scores))

x = [1,2,3,4,5,6,7,8,9,10]
y = d_t_scores
plt.subplot(1, 2, 1)
plt.plot(x,y)
plt.xlabel("k fold iteration")
plt.ylabel("accuracy")
plt.title('Decision Tree K Fold')
y2 = d_l_scores
plt.subplot(1, 2, 2)
plt.plot(x,y2)
plt.xlabel("k fold iteration")
plt.ylabel("accuracy")
plt.title('Deep Learning K Fold')
plt.show()

"""
sum_eng = 0
sum_non = 0
sum_amb = 0
sum_cod = 0

for i in range (len(labels)):
	if(labels[i] == 0):
		sum_eng += 1
	elif(labels[i] == 1):
		sum_amb += 1
	elif(labels[i] == 2):
		sum_non += 1
	elif(labels[i] == 3):
		sum_cod += 1
plt.bar(['English', 'Ambiguous', 'Non-English', 'Code-Switched'], [sum_eng, sum_amb, sum_non, sum_cod])
plt.show()
"""