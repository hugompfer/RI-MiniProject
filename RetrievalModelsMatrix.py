import numpy as np

class RetrievalModelsMatrix:

    def __init__(self, tf, vectorizer, mius, lmds):
        self.vectorizer = vectorizer
        self.tf = tf
        self.term_coll_freq = np.sum(tf, axis=0)
        self.docLen = np.sum(tf, axis=1) + 1
        self.term_doc_freq = np.sum(tf != 0, axis=0)

        ## VSM statistics
        idf_value = np.log(np.size(tf, axis = 0) / self.term_doc_freq) 
        self.idf = (idf_value > 0.01) * idf_value
        self.tfidf = np.array(tf * self.idf)

        self.docNorms = np.sqrt(np.sum(np.power(self.tfidf, 2), axis=1))

        ## LMD statistics
        self.collection_size = np.sum(self.docLen)
        self.term_coll_freq_prob = np.divide(self.term_coll_freq , self.collection_size)
        self.term_doc_freq_prob = self.tf/np.array(self.docLen)[:, None]

        self.lmd_matrixs = list(map(self.__lmd_function, mius))

        ## LMJM statistics
        self.lmjm_matrixs = list(map(self.__lmjm_function, lmds))
        
        ## BM25 statistics

    def __create_matrix_dic(self, param, matrix):
        return {
            "param": param,
            "matrix": matrix
        }

    def __lmd_function(self, miu):
        matrix = (self.tf + miu * self.term_coll_freq_prob) / (np.reshape(self.docLen, [np.size(self.docLen), 1]) + miu)
        return self.__create_matrix_dic(miu, matrix)

    def __lmjm_function(self, lmd):
        matrix = lmd * self.term_doc_freq_prob + (1 - lmd) * self.term_coll_freq_prob
        return self.__create_matrix_dic(lmd, matrix)

    def score_vsm(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        query_norm = np.sqrt(np.sum(np.power(query_vector, 2), axis=1))

        doc_scores = np.dot(query_vector, self.tfidf.T) / (0.0001 + self.docNorms * query_norm)

        return doc_scores

    def score_lmd(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        doc_scores = map(lambda matrix_dic: {
            "param": matrix_dic["param"],
            "result": np.array(np.prod(matrix_dic["matrix"] ** query_vector, axis=1), dtype=np.ndarray)
        }, self.lmd_matrixs)

        return list(doc_scores)

    def score_lmjm(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        doc_scores = map(lambda matrix_dic: {
            "param": matrix_dic["param"],
            "result": np.prod(matrix_dic["matrix"] ** query_vector, axis=1)
        }, self.lmjm_matrixs)

        return list(doc_scores)

    def score_bm25(self, query):
        return 0

    def scoreRM3(self, query):
        return 0

