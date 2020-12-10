import string
import csv
import re
import numpy as np
import os
import pickle


def idf(docs, total_docs):
    DF = len(docs)
    return 1 + np.log(len(total_docs) / DF)


table = str.maketrans('', '', string.punctuation)


def tf_idf(mydict, total_docs):
    terms = mydict.keys()
    TF = np.zeros((len(terms), len(total_docs)))
    DF = []
    for i, term in enumerate(terms):
        for doc in mydict[term]:
            song_name = doc.split('/')[-1]
            j = total_docs.index(song_name)
            content = open(doc, 'r', encoding='utf-8').read()

            words = list(map(
                lambda x: x[:-1] if x[-1] in [',', '!', '?', '.'] else x, content.lower().split()))
            content = [w.translate(table) for w in words]
            TF[i, j] = content.count(term)
        DF.append(len(mydict[term]))

    TF = np.array(TF[1:])
    DF = np.array(DF[1:])
    IDF = 1 + np.log(len(total_docs) / DF)
    IDF = np.array([IDF]).T
    TF = TF / np.sum(TF, axis=0)
    return TF * IDF


def main():
    mydict = {}
    # read csv file to get {word : files path}
    with open('./utils/index-file.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            mydict[row[0]] = row[1].split(" _ ")

    total_docs = os.listdir('./data/lyrics')
    TF_IDF = tf_idf(mydict, total_docs)
    pickle.dump(TF_IDF, open('tf_idf.pk', 'wb'))
    pk = open('tf_idf.pk', 'rb')
    data = pickle.load(pk)
    print(data)


main()
