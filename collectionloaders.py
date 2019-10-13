import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import simpleparser
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


class CranfieldTestBed:

    def __init__(self):

        # fields: docid, title, author, journal, abstract
        self.corpus_cranfield = pd.read_csv('./corpus/cran.all.1400', sep=';')
        self.num_docs = np.size(self.corpus_cranfield, axis = 0)

        # fields: qid, query
        self.queries_cranfield = pd.read_csv('./corpus/cran.qry', sep=';')
        self.num_queries = np.size(self.queries_cranfield, axis = 0)

        # fields: qid, docid, rel
        self.relevance_judgments_cranfield = pd.read_csv('./corpus/cranqrel', sep=' ')

        #qid = queries_cranfield['qid']
        self.queries = self.queries_cranfield['query']

        print('Number of documents: ', self.num_docs)
        print('Number of queries: ', self.num_queries)
        
        return


    def eval(self, scores, this_qid):
        idx_rel_docs = self.relevance_judgments_cranfield.loc[self.relevance_judgments_cranfield['qid'] == (this_qid)]
        #print(idx_rel_docs)

        query_rel_docs = idx_rel_docs['docid']-1

        relv_judg_list = idx_rel_docs['rel']
        #print(relv_judg_list)
        
        rank = np.argsort(scores, axis = None)
        top10 = rank[-10:]
        true_pos= np.intersect1d(top10,query_rel_docs)
        p10 = np.size(true_pos) / 10

        relev_judg_vector = np.zeros((self.num_docs,1))
        relev_judg_vector[query_rel_docs,0] = (relv_judg_list>0)

        average_precision = average_precision_score(relev_judg_vector, scores.T)

        precision, recall, thresholds = precision_recall_curve(relev_judg_vector, scores.T)

        precision_interpolated = np.maximum.accumulate(precision)
        recall_11point = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        precision_11point = np.interp(recall_11point, np.flip(recall), np.flip(precision_interpolated))

        if False:
            plt.plot(recall, precision, color='b', alpha=1) # Raw precision-recall
            plt.plot(recall, precision_interpolated, color='r', alpha=1) # Interpolated precision-recall
            plt.plot(recall_11point, precision_11point, color='g', alpha=1) # 11-point interpolated precision-recall

        return [average_precision, precision_11point, recall_11point, p10]
