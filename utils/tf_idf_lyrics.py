'''
output: list of songs names
'''
import string
import csv
import re
import numpy as np
import os
import time
import pickle
from scipy.spatial import distance


table = str.maketrans('', '', string.punctuation)


def idf(docs, total_docs):
    DF = len(docs)
    return 1 + np.log(len(total_docs) / DF)


def linked(query, songs_names):
    '''
    query: list
    songs_names = list of songs's names
    '''
    linked_points = []
    # counter = []

    for song_name in songs_names:
        variance = 1
        plinked_points = 0
        content = open('./data/lyrics/' + song_name,
                       'r', encoding='utf-8').read()
        content = content.translate(table)
        for it in range(len(query) - 1):
            temp = plinked_points
            indexes = re.finditer("\\b(?i)" + query[it] + "\\b", content)
            for index in indexes:
                start_pos = index.end(0)
                for i in range(it + 1, len(query)):
                    if content[start_pos + 1: start_pos + 1 + len(query[i])].lower() == query[i]:
                        start_pos += len(query[i]) + 1
                        plinked_points += i * i
                    else:
                        break
            if plinked_points > temp:
                variance += 1
        linked_points.append(plinked_points * variance * variance)
    return linked_points


def qtf_idf(mydict, query):
    terms = mydict.keys()
    TF = np.zeros((len(terms), 1))
    for i, term in enumerate(terms):
        # for j, word in enumerate(query):
        TF[i] = query.count(term)
    TF = TF[1:]
    IDF = idf(mydict[term], query)
    TF = TF / np.sum(TF, axis=0)
    return TF * IDF


def main(query):
    mydict = {}
    # read csv file to get {word : files path}
    with open('./utils/index-file.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            mydict[row[0]] = row[1].split(" _ ")

    # preprocess query
    words = list(map(
        lambda x: x[:-1] if x[-1] in [',', '!', '?', '.'] else x, query.lower().split()))

    query = [w.translate(table) for w in words]
    total_docs = os.listdir('./data/lyrics')
    pk = open('tf_idf.pk', 'rb')
    TF_IDF = pickle.load(pk)
    qTF_IDF = qtf_idf(mydict, query)

    # TF_IDF = TF_IDF.T
    # dists = []
    # for ele in TF_IDF:
    #     dists.append(distance.cosine(qTF_IDF, ele))
    # dists = np.array(dists)
    dists = np.linalg.norm(qTF_IDF - TF_IDF, axis=0)
    rank = np.argsort(dists)
    topK = 500
    res = []
    for i in range(topK):
        res.append(total_docs[rank[i]])

    # ranking
    linked_res = linked(query, res)
    rank = np.argsort(linked_res)[::-1][:topK]
    final_res = []
    for i in range(10):
        final_res.append(res[rank[i]])
    return final_res


start = time.time()
query = "Constricted my grasp"
qlst = query.split()
songs = main(query)

for song in songs:
    count = 0
    with open('./data/lyrics/' + song, 'r', encoding='utf-8') as f:
        l = f.read()
        for q in qlst:
            count += l.count(q)
    print(song, str(count))
print(time.time() - start)
