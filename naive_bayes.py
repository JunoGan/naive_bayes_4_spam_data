
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv('spam.csv', usecols=[0,1], encoding='latin-1')
df.columns = ['label', 'body']

df['label'] = df['label'].replace(['ham', 'spam'], [0, 1])
data = df.values

# split into training and testing data
train_data, test_data = train_test_split(data, train_size = 0.75)

# convert into n grams
n = 2

def ngrams(n, text):
    text = text.split(' ')
    grams = []
    for i in range(len(text) - n + 1):
        gram = ' '.join(text[i : i + n])
        grams.append(gram)
    return grams

train_data = [[item[0], ngrams(n, item[1])] for item in train_data]
test_data = [[item[0], ngrams(n, item[1])] for item in test_data]

# count unique n grams in training data
flattened = [gram for message in train_data for gram in message[1]]
unique = len(set(flattened))

# init dicts
trainPositive = {}
trainNegative = {}
    
# counters
posGramCount = 0
negGramCount = 0
spamCount = 0
# priors
pA = 0
pNotA = 0

# train
for item in train_data:
    label = item[0]
    grams = item[1]
    if label == 1:
        spamCount += 1
    for gram in grams:
        if label == 1:
            trainPositive[gram] = trainPositive.get(gram, 0) + 1
            posGramCount += 1
        else:
            trainNegative[gram] = trainNegative.get(gram, 0) + 1
            negGramCount += 1
            
pA = spamCount / float(len(train_data))
pNotA = 1.0 - pA

#evaluate on testing data
def classify(text):
    isSpam = pA * conditionalText(text, 1)
    notSpam = pNotA * conditionalText(text, 0)
    if (isSpam > notSpam):
        return 1
    else:
        return 0
    
def conditionalText(grams, label):
    result = 1.0
    for ngram in grams:
        result *= conditionalNgram(ngram, label)
    return result
#Laplace Smoothing for the words not present in the training set
#gives the conditional probability p(text | spam/notSpam) with smoothing
def conditionalNgram(ngram, label):
    alpha = 1.0
    if label == 1:
        return ((trainPositive.get(ngram, 0) + alpha) /
                    float(posGramCount + alpha * unique))
    else:
        return ((trainNegative.get(ngram, 0) + alpha) /
                    float(negGramCount + alpha * unique))
    

results = []
for test in test_data:
    label = test[0]
    text = test[1]
    ruling = classify(text)
    if ruling == label:
        results.append(1) 
    else:
        results.append(0) 
print("Evaluated {} test cases. {:.2f}% Accuracy".format(len(results), 100.0 * sum(results) / float(len(results))))



















