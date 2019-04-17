from surprise import Reader, Dataset
from surprise import KNNWithMeans
from surprise import accuracy
from surprise.model_selection import train_test_split, cross_validate

class MyAlgo():

    def __init__(self, rating_data=''):
        if rating_data:
            reader = Reader(line_format='user item rating timestamp', sep=',')
            self.ratings = Dataset.load_from_file(rating_data, reader)
            self.trainset, self.testset = train_test_split(self.ratings, test_size=0.25)
            self.sim_options = {'name': 'cosine','user_based': False}
            
    
    def setK(self, k_value):
        algo = KNNWithMeans(k=k_value, sim_options=self.sim_options)
        self.algo = algo
        self.algo.fit(self.trainset)


    def findBestK(self):
        for k_value in [2, 3, 5, 10, 20, 30, 40]:
            print('K = {}'.format(k_value))
            algo = KNNWithMeans(k=k_value, sim_options=self.sim_options)
            cross_validate(algo, self.ratings, measures=['RMSE', 'MAE'], cv=3, verbose=True)
            print('\n\n')