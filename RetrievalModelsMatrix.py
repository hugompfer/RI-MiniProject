import numpy as np

class RetrievalModelsMatrix:

    def __init__(self, tf, vectorizer, mius, lmds, term_thresholds):
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
        self.mius = mius
        self.collection_size = np.sum(self.docLen)
        self.term_coll_freq_prob = np.divide(self.term_coll_freq , self.collection_size)
        self.term_doc_freq_prob = self.tf/np.array(self.docLen)[:, None]

        self.lmd_matrixs = list(map(self.__lmd_function, mius))

        ## LMJM statistics
        self.lmjm_matrixs = list(map(self.__lmjm_function, lmds))

        ## RM3
        self.doc_threshold = 3
        self.term_thresholds = term_thresholds

    def __create_matrix_dic(self, param, matrix):
        return {
            "param": param,
            "matrix": matrix
        }

    def __calculate_lmd(self, miu):
        return (self.tf + miu * self.term_coll_freq_prob) / (np.reshape(self.docLen, [np.size(self.docLen), 1]) + miu)

    def __lmd_function(self, miu):
        return self.__create_matrix_dic(miu, self.__calculate_lmd(miu))

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


    def __calculateRM3(self, miu, term_threshold, query_vector):
        lmd_result = self.__calculate_lmd(miu)
        ranks = np.prod(lmd_result ** query_vector, axis=1)
        sorted_ranks = np.sort(ranks)
        threshold = sorted_ranks[self.doc_threshold]

        top_documents = ranks * (ranks > threshold)
        top_term_query_prob = self.term_doc_freq_prob * np.reshape(top_documents, [-1, 1])
        top_doc_term_query_prob = np.sum(top_term_query_prob, axis=0)
        term_query_prob = query_vector / np.sum(query_vector, axis=1)

        final_query = ((1 - term_threshold) * term_query_prob + term_threshold * top_doc_term_query_prob)[0]

        sorted_final_query = np.sort(final_query)
        final_query_threshold = sorted_final_query[query_vector.shape[0] + 5]
        best_terms = sorted_final_query * (sorted_final_query > final_query_threshold)

        return np.prod(lmd_result ** best_terms, axis=1)

    def scoreRM3(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        results = dict()

        for miu in self.mius:
            results[miu] = dict()
            for tt in self.term_thresholds:
                results[miu][tt] = self.__calculateRM3(miu, tt, query_vector)

        return results
