import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import simpleparser as parser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import collectionloaders
import RetrievalModelsMatrix as models


class Statistics:

    def __init__(self, cranfield):
        self.verbose = True
        self.bigrams = True
        self.cranfield = cranfield
        if not self.bigrams:
            self.vectorizer = CountVectorizer()
        else:
            self.vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b',
                                         min_df=1, stop_words={'the', 'is'})

        # Tokenize, stem and remove stop words
        self.corpus = parser.stemCorpus(cranfield.corpus_cranfield['abstract'])

        # Create the model
        # Compute the term frequencies matrix and the model statistics
        self.tf_cranfield = self.vectorizer.fit_transform(self.corpus).toarray()
        self.models = models.RetrievalModelsMatrix(self.tf_cranfield, self.vectorizer)

    def calculate(self):
        i = 1
        map_vsm = 0
        precision_vsm = []
        for query in self.cranfield.queries:
            # Parse the query and compute the document scores
            scores = self.models.score_vsm(parser.stemSentence(query))

            # Do the evaluation
            [average_precision, precision, recall, thresholds] = self.cranfield.eval(scores, i)
            map_vsm = map_vsm + average_precision
            precision_vsm.append(precision)

            # Some messages...
            if self.verbose:
                plt.plot(recall, precision, color='silver', alpha=0.1)
                print('qid =', i, 'VSM     AP=', average_precision)

            i = i + 1

        map_vsm = map_vsm / self.cranfield.num_queries

        return precision_vsm, recall, map_vsm

    def plot_precision_recall(self, precision_vsm, recall, map_vsm):
        mean_precision = np.mean(precision_vsm, axis=0)
        std_precision = np.std(precision_vsm, axis=0)

        plt.plot(recall, np.mean(precision_vsm, axis=0), color='b', alpha=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.fill_between(recall,
                         mean_precision - std_precision,
                         mean_precision + std_precision, facecolor='b', alpha=0.1)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall (MAP={0:0.2f})'.format(map_vsm))
        plt.savefig('results/prec-recall.png', dpi=100)