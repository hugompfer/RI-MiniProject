import collectionloaders
import Statistics
import pickle

class Data:

    def __init__(self):
        self.cranfield = collectionloaders.CranfieldTestBed()
        self.statistics = Statistics.Statistics(self.cranfield)

    def calculate(self):
        self.results = self.statistics.calculate()

    def get_data_table(self):
        return {
            'Retrievel Model': ['VSM', 'LMD @ 0.1','LMD @ 0.99', 'LMJM @ 100', 'LMJM @ 1000', 'BM25'],
            'P10': ['0', '0', '0', '0', '0', '0'],
            'MAP': [
                self.results['uni']['VSM']['map_vsm'],
                self.results['uni']['LMD'][0.1]['map_vsm'],
                self.results['uni']['LMD'][0.99]['map_vsm'],
                self.results['uni']['LMJM'][100.0]['map_vsm'],
                self.results['uni']['LMJM'][1000.0]['map_vsm'],
                '--'
                    ]
        }

    def save_data(self):
        with open('data.pickle', 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self):
        with open('data.pickle', 'rb') as handle:
            self.results = pickle.load(handle)
