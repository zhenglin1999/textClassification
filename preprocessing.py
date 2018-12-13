import string
import nltk
import math
import nltk.corpus
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import *


def load_dataset():
    file = open('sample.txt', 'r')
    dic = {}
    for line in file:
        if not len(line.strip()):
            continue
        tokens = line.strip().split(':')
        if tokens[0] not in dic:
            dic[tokens[0]] = [tokens[1]]
        else:
            dic[tokens[0]].append(tokens[1])
    dic = pd.DataFrame(dic)
    print(dic)
    return dic


def tf(word, count):
    return count[word] / sum(count.values())


def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)


def idf(word, count_list):
    return math.log(len(count_list) / (1 + n_containing(word, count_list)))


def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


def get_refined_token(text):
    stemmer = PorterStemmer()
    res = []
    lower = text.lower()
    punctuation_removed = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lower.translate(punctuation_removed)
    tokens = nltk.word_tokenize(no_punctuation)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    for item in tokens:
        res.append(stemmer.stem(item))
    return res


if __name__ == '__main__':
    dic = load_dataset()
    countlist = dic['review/text']
    countlist = [Counter(get_refined_token(w)) for w in countlist]
    for i, count in enumerate(countlist):
        print("Top words in document {}".format(i + 1))
        scores = {word: tfidf(word, count, countlist) for word in count}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:3]:
            print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
