import numpy as np

class RetrievalModelsInvertedIndex:

    def __init__(self, tf, vectorizer):

        self.vectorizer = vectorizer
        self.features = vectorizer.get_feature_names()
        self.docNorms = np.zeros(1, np.size(tf,1))
        self.idf = []

        i = 0
        self.inverted_index = dict()
        for token in self.features:
            print("==== Creating the posting list for token \"", token, "\"")
            docs_with_token = np.where(tf[:, i] != 0)
            len = np.size(docs_with_token, 1)

            postings_matrix = np.concatenate([tf[docs_with_token, i], docs_with_token])
            postings_list = list(map(tuple, postings_matrix.T))
            self.inverted_index[token] = postings_list

            print(postings_list)
            i = i + 1
            self.docNorms = []

        self.idf = []

    def scoreVSM(self, query):

        return doc_scores


    def scoreLMD(self, query):
        return 0


    def scoreLMJM(self, query):
        return 0


    def scoreBM25(self, query):
        return 0


    def scoreRM3(self, query):
        return 0

