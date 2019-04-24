from collections import defaultdict
import pandas as pd
from surprise import Reader, Dataset
from surprise import KNNWithMeans
from surprise import accuracy
from surprise.model_selection import train_test_split, cross_validate


class MyAlgo02():
    def __init__(self, rating_data=''):
        if rating_data:
            reader = Reader(line_format='user item rating timestamp', sep=',')
            self.ratings = Dataset.load_from_file(rating_data, reader)
#             self.trainset, self.testset = train_test_split(self.ratings, test_size=0.25)
            self.trainset = self.ratings.build_full_trainset()
            self.sim_options = {'name': 'cosine','user_based': False}

        
    def set_k(self, k_value):
        algo = KNNWithMeans(k=k_value, sim_options=self.sim_options)
        self.algo = algo
        self.algo.fit(self.trainset)
        
        
    def find_best_k(self):
        for k_value in [2, 3, 5, 10, 20, 30, 40]:
            print('K = {}'.format(k_value))
            algo = KNNWithMeans(k=k_value, sim_options=self.sim_options)
            cross_validate(algo, self.ratings, measures=['RMSE', 'MAE'], cv=3, verbose=True)
            print('\n\n')
        
        
    def get_top_n(predictions, n=10):
        '''Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.

        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        '''

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n


    def predict_ratings(self):
        # Predict ratings for all pairs (u, i) that are NOT in the training set.
        testset = self.trainset.build_anti_testset()
        self.testset = testset
        predictions = self.algo.test(self.testset)
        self.predictions = predictions
        
        
    def recs_for_user(self, uid):
        user_filtered = list(filter(lambda x: x.uid == str(uid), self.predictions))
        
        print(len(user_filtered))
        
        top_n = self.get_top_n(predictions=user_filtered, n=10)
        
        return top_n




bla = MyAlgo02('ml-latest-small/ratings.csv')
print(bla.sim_options)

print("Setting K...")
bla.set_k(k_value=10)

print("Predicting ratings for movies not rated yet...")
bla.predict_ratings()
len(bla.predictions)

print("Recommending to user: ")
my_recs = bla.recs_for_user(uid=2)
my_recs