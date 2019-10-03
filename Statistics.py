import numpy as np
import matplotlib.pyplot as plt
import simpleparser as parser
from sklearn.feature_extraction.text import CountVectorizer
import RetrievalModelsMatrix as models
import json

class Statistics:

    def __init__(self, cranfield, bigrams):
        self.bigrams = bigrams
        self.cranfield = cranfield
        self.vectorizer = self.__init_vectorizer()
        self.models_to_calculate = self.__load_settings()
        # Tokenize, stem and remove stop words
        self.corpus = parser.stemCorpus(cranfield.corpus_cranfield['abstract'])

        # Create the model
        # Compute the term frequencies matrix and the model statistics
        self.tf_cranfield = self.vectorizer.fit_transform(self.corpus).toarray()
        self.models = models.RetrievalModelsMatrix(self.tf_cranfield, self.vectorizer)
        self.scoreModels = {
            "VSM": self.models.score_vsm,
            "LMD": self.models.score_lmd,
            "LMJM": self.models.score_lmjm,
            "BM25": self.models.score_bm25
        }

    def __load_settings(self):
        f = open('settings.json')
        data = json.load(f)
        f.close()
        return data

    def __init_vectorizer(self):
        if not self.bigrams:
            return CountVectorizer()
        else:
            return CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b',
                                   min_df=1, stop_words={'the', 'is'})

    def calculate(self):
        self.models_to_calculate = self.__load_settings()

        for model in self.models_to_calculate.keys():
            model = model.upper()
            i = 1
            precision_vsm = []
            average_precisions = []
            map_vsm = 0

            for query in self.cranfield.queries:
                # Parse the query and compute the document scores
                stem_query = parser.stemSentence(query)
                scores = self.get_score(model, stem_query)

                if isinstance(scores,dict):
                    for param in scores.keys():
                        [average_precision, precision, recall, thresholds] = self.cranfield.eval(scores.keys()[param], i)


                # Do the evaluation
                [average_precision, precision, recall, thresholds] = self.cranfield.eval(scores, i)
                map_vsm = map_vsm + average_precision
                precision_vsm.append(precision)

                average_precisions.append([i, average_precision])
                i = i + 1

            map_vsm = map_vsm / self.cranfield.num_queries

        return precision_vsm, recall, map_vsm, average_precisions

    def get_score(self, model, query):
        scores = {}
        if model == 'VSM':
            return self.scoreModels[model](query)
        elif model == 'LMD' or model == 'LMJM':
            for param in self.models_to_calculate[model].params:
                scores[param] = self.scoreModels[model](query, param)
        else:
            b_params, k_params = self.models_to_calculate[model]["params"].keys()
            for b in b_params:
                for k in k_params:
                    scores[b+"_"+k] = self.scoreModels[model](query, k, b)
        return scores

    def plot_average_precision_table(self, average_precisions):
        columns = ('Query Id', 'Average precision')
        table = plt.table(cellText=average_precisions,
                              colLabels=columns,
                              loc='center')
        table.set_fontsize(14)
        table.scale(1, 2)
        plt.show()

    def plot_precision_recall(self, precision_vsm, recall, map_vsm):
        for precision in precision_vsm:
            plt.subplot(2, 1, 1)
            plt.plot(recall, precision, color='silver', alpha=0.1)

        plt.subplot(2, 1, 2)
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
