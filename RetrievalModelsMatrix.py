import numpy as np


class RetrievalModelsMatrix:

    def __init__(self, tf, vectorizer):
        self.vectorizer = vectorizer
        self.tf = tf

        ## VSM statistics
        self.term_doc_freq = np.sum(tf != 0, axis=0)#term collection freq
        self.term_coll_freq = np.sum(tf, axis=0) #term collection freq
        self.docLen = np.sum(tf, axis=1) #document size

        idf_value = np.log(np.size(tf, axis = 0) / self.term_doc_freq) 
        self.idf = (idf_value > 0.01) * idf_value
        self.tfidf = np.array(tf * self.idf)

        self.docNorms = np.sqrt(np.sum(np.power(self.tfidf, 2), axis=1))

        ## LMD statistics
        self.collection_size = np.sum(self.docLen)
        self.term_coll_freq_prob = np.divide(self.term_coll_freq / self.collection_size)
        self.term_doc_freq_prob = self.tf/np.array(self.docLen)[:, None]

        ## LMJM statistics

        
        ## BM25 statistics


    def score_vsm(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        query_norm = np.sqrt(np.sum(np.power(query_vector, 2), axis=1))

        doc_scores = np.dot(query_vector, self.tfidf.T) / (0.0001 + self.docNorms * query_norm)

        return doc_scores

    def score_lmd(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        indexes = np.where(query_vector[0] != 0)[0]
        
        return doc_scores

    def score_lmjm(self, query):
        return 0

    def score_bm25(self, query):
        return 0

    def scoreRM3(self, query):
        return 0

