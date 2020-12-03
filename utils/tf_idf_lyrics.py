import string
import csv
import re
import numpy as np
import os


def idf(docs, total_docs):
    DF = len(docs)
    return 1 + np.log(len(total_docs) / DF)


table = str.maketrans('', '', string.punctuation)


def calc_TF_IDF(vocab, mydict, content):
    for word in vocab:
        for doc in mydict[word]:
            pass


def tf_idf(mydict, total_docs):
    terms = mydict.keys()
    TF = np.zeros((len(terms), len(total_docs)))
    IDF = []
    for i, term in enumerate(terms):
        for doc in mydict[term]:
            song_name = doc.split('/')[-1]
            j = total_docs.index(song_name)
            doc = open(doc, 'r', encoding='utf-8').read()

            words = list(map(
                lambda x: x[:-1] if x[-1] in [',', '!', '?', '.'] else x, doc.lower().split()))
            doc = [w.translate(table) for w in words]
            TF[i, j] = doc.count(term)
        IDF.append(idf(mydict[term], total_docs))

    IDF = np.array([IDF]).T
    TF = TF / np.sum(TF, axis=0)
    return TF * IDF


def qtf_idf(mydict, query):
    terms = mydict.keys()
    TF = np.zeros((len(terms), 1))
    for i, term in enumerate(terms):
        # for j, word in enumerate(query):
        TF[i] = query.count(term)
    IDF = idf(mydict[term], query)
    TF = TF / np.sum(TF, axis=0)
    return TF * IDF


def main(query):
    mydict = {}
    # read csv file to get {word : files path}
    with open('./utils/index-file.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            mydict[row[0]] = row[1].split(", ")

    # preprocess query
    # capital words
    # query = query.lower()
    words = list(map(
        lambda x: x[:-1] if x[-1] in [',', '!', '?', '.'] else x, query.lower().split()))

    # stop words
    # words = query.split()

    table = str.maketrans('', '', string.punctuation)
    query = [w.translate(table) for w in words]

    # start retrieval
    postings = []
    for word in query:
        try:
            postings.append(set(mydict[word]))
        except:
            continue

    if len(postings) == 0:
        return [None]

    songs_name = set.intersection(*postings)

    # rankings
    mydict.pop('\ufeffroad')
    total_docs = os.listdir('./data/lyrics')
    TF_IDF = tf_idf(mydict, total_docs)
    qTF_IDF = qtf_idf(mydict, query)

    dists = np.linalg.norm(TF_IDF - qTF_IDF, axis=0)
    rank = np.argsort(dists)
    topK = 10
    for i in range(topK):
        print('Van ban gan thu ', i+1, ' la: ',
              ' '.join(total_docs[rank[i]]))

    # results = []
    # # get index of query in found songs
    # for song_name in songs_name:
    #     result = []
    #     result.append(song_name.split('/')[-1].split('.')[0])
    #     index_arr = [0]
    #     lyric = []
    #     with open(song_name, 'r', encoding='utf-8') as f:
    #         song_lyric = f.read()
    #         for word in query:
    #             indexes = re.finditer("\\b(?i)" + word + "\\b", song_lyric)
    #             for index in indexes:
    #                 # print(index.start(0), index.end(0))
    #                 index_arr.append(index.start(0))
    #                 index_arr.append(index.end(0))
    #     index_arr.append(len(song_lyric))
    #     index_arr = sorted(index_arr)
    #     for i in range(len(index_arr)-1):
    #         temp = song_lyric[index_arr[i]:index_arr[i+1]]
    #         temp = temp.replace('\n', '<br>')
    #         lyric.append(temp)
    #     lyric.append(song_lyric[index_arr[len(index_arr)-1]:])
    #     result.append(lyric)
    #     results.append(result)
    # return results


main("Galway")
# Van ban gan thu  1  la:  G a l w a y   G i r l   -   E d   S h e e r a n . t x t
# Van ban gan thu  2  la:  B à i   N à y   C h i l l   P h ế t   -   Đ e n _ M I N . t x t
# Van ban gan thu  3  la:  Â m   T h ầ m   B ê n   E m   -   S ơ n   T ù n g   M - T P . t x t
# Van ban gan thu  4  la:  T ì m   ( L o s t )   -   M I N _ M R . A . t x t
# Van ban gan thu  5  la:  M ộ t   T r i ệ u   N ă m   Á n h   S á n g   -   V ũ   C á t   T ư ờ n g . t x t
# Van ban gan thu  6  la:  Đ ừ n g   X i n   L ỗ i   N ữ a   ( D o n ' t   S a y   S o r r y )   -   E R I K _ M I N . t x t
# Van ban gan thu  7  la:  N ă m   Ấ y   -   Đ ứ c   P h ú c . t x t
# Van ban gan thu  8  la:  M ặ t   T r ờ i   C ủ a   E m   -   P h ư ơ n g   L y _ J u s t a T e e   . t x t
# Van ban gan thu  9  la:  V ẫ n   N h ớ   -   T u ấ n   H ư n g . t x t
# Van ban gan thu  10  la:  N ơ i   N à y   C ó   A n h   -   S ơ n   T ù n g   M - T P . t x t

main("Hallelujah")
# Van ban gan thu  1  la:  S u p e r m a r k e t   F l o w e r s   -   E d   S h e e r a n . t x t
# Van ban gan thu  2  la:  B à i   N à y   C h i l l   P h ế t   -   Đ e n _ M I N . t x t
# Van ban gan thu  3  la:  Â m   T h ầ m   B ê n   E m   -   S ơ n   T ù n g   M - T P . t x t
# Van ban gan thu  4  la:  T ì m   ( L o s t )   -   M I N _ M R . A . t x t
# Van ban gan thu  5  la:  M ộ t   T r i ệ u   N ă m   Á n h   S á n g   -   V ũ   C á t   T ư ờ n g . t x t
# Van ban gan thu  6  la:  Đ ừ n g   X i n   L ỗ i   N ữ a   ( D o n ' t   S a y   S o r r y )   -   E R I K _ M I N . t x t
# Van ban gan thu  7  la:  N ă m   Ấ y   -   Đ ứ c   P h ú c . t x t
# Van ban gan thu  8  la:  M ặ t   T r ờ i   C ủ a   E m   -   P h ư ơ n g   L y _ J u s t a T e e   . t x t
# Van ban gan thu  9  la:  V ẫ n   N h ớ   -   T u ấ n   H ư n g . t x t
# Van ban gan thu  10  la:  N ơ i   N à y   C ó   A n h   -   S ơ n   T ù n g   M - T P . t x t