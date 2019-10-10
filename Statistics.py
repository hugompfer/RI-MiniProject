import numpy as np
import matplotlib.pyplot as plt
import simpleparser as parser
from sklearn.feature_extraction.text import CountVectorizer
import RetrievalModelsMatrix as models


class Statistics:

    def __init__(self, cranfield):
        self.models_names = ["VSM", "LMD", "LMJM", "RM3"]
        self.cranfield = cranfield
        self.vectorizer_unigram = CountVectorizer()
        self.vectorizer_bigram = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b',
                                                 min_df=1, stop_words={'the', 'is'})

        # Tokenize, stem and remove stop words
        self.corpus = parser.stemCorpus(cranfield.corpus_cranfield['abstract'])

        # Create the model
        # Compute the term frequencies matrix and the model statistics
        self.tf_cranfield_uni = self.vectorizer_unigram.fit_transform(self.corpus).toarray()
        self.tf_cranfield_big = self.vectorizer_bigram.fit_transform(self.corpus).toarray()

        colection_size = np.sum(np.sum(self.tf_cranfield_uni, axis=1)) / cranfield.num_docs
        self.mius = [colection_size * 0.8, colection_size * 0.9, colection_size, colection_size * 1.1,
                     colection_size * 1.2]
        self.lmds = np.linspace(0.1, 0.9, num=5)
        self.term_thresholds = np.linspace(0.1, 0.9, num=5)

        self.models_uni = models.RetrievalModelsMatrix(self.tf_cranfield_uni, self.vectorizer_unigram, self.mius, self.lmds, self.term_thresholds)
        self.models_big = models.RetrievalModelsMatrix(self.tf_cranfield_big, self.vectorizer_bigram, self.mius, self.lmds, self.term_thresholds)

    def calculate_models_scores(self, models, query):
        return {
            "VSM": models.score_vsm(query),
            "LMD": models.score_lmd(query),
            "LMJM": models.score_lmjm(query),
            "RM3": models.scoreRM3(query)
        }

    def __create_statistics_dic(self):
        return {
            "precision_vsm": [],
            "recall": [],
            "map_vsm": 0,
            "precision@10": 0
        }

    def __create_param_dic(self, results_dic, param):
        results_dic[param] = self.__create_statistics_dic()

    def create_result_dic(self):
        results = {
            "uni": dict(),
            "big": dict()
        }

        for model_score in self.models_names:
            if model_score == "VSM":
                self.__create_param_dic(results["uni"], model_score)
                self.__create_param_dic(results["big"], model_score)
            else:
                results["uni"][model_score] = dict()
                results["big"][model_score] = dict()
                if model_score == "LMD" or model_score == "LMJM":
                    params = self.mius if model_score == "LMD" else self.lmds
                    for param in params:
                        self.__create_param_dic(results["uni"][model_score], param)
                        self.__create_param_dic(results["big"][model_score], param)
                if model_score == "RM3":
                    for miu in self.mius:
                        results["uni"][model_score][miu] = dict()
                        for tt in self.term_thresholds:
                            self.__create_param_dic(results["uni"][model_score][miu], tt)
        return results

    def __update_statistic(self, average_precision, precision, recall, precision_10, result):
        result["precision_vsm"].append(precision)
        result["map_vsm"] = result["map_vsm"] + average_precision
        result["recall"] = recall
        result["precision@10"] = precision_10

    def __update_statistics_scores(self, scores, statistic_results, i):
        models = scores.keys()
        cranfield_size = self.cranfield.num_queries
        for key in models:
            i = 1
            model_score = scores[key]
            if key == "VSM":
                [average_precision, precision, recall, precision_10] = self.cranfield.eval(model_score, i)
                self.__update_statistic(average_precision, precision, recall, precision_10, statistic_results[key])
                if i == cranfield_size:
                    statistic_results[key]["map_vsm"] = statistic_results[key]["map_vsm"] / cranfield_size
            elif key == "LMD" or key == "LMJM":
                for score_dic in model_score:
                    [average_precision, precision, recall, precision_10] = self.cranfield.eval(score_dic["result"], i)
                    self.__update_statistic(average_precision, precision, recall, precision_10, statistic_results[key][score_dic["param"]])
                    if i == cranfield_size:
                        statistic_results[key][score_dic["param"]]["map_vsm"] = statistic_results[key][score_dic["param"]]["map_vsm"] / cranfield_size
            elif key == "RM3":
                for miu in model_score.keys():
                    miu_dic = model_score[miu]
                    for tt in miu_dic.keys():
                        res = miu_dic[tt]
                        [average_precision, precision, recall, precision_10] = self.cranfield.eval(res, i)
                        self.__update_statistic(average_precision, precision, recall, precision_10, statistic_results[key][miu][tt])
                        if i == cranfield_size:
                            statistic_results[key][miu][tt]["map_vsm"] = statistic_results[key][miu][tt]["map_vsm"] / cranfield_size

    def calculate(self):
        i = 1
        results = self.create_result_dic()

        for query in self.cranfield.queries:
            # Parse the query and compute the document scores
            stem_query = parser.stemSentence(query)

            scores_uni = self.calculate_models_scores(self.models_uni, stem_query)
            score_big = self.calculate_models_scores(self.models_big, stem_query)

            self.__update_statistics_scores(scores_uni, results["uni"], i)
            self.__update_statistics_scores(score_big, results["big"], i)

            i = i + 1

        return results

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
