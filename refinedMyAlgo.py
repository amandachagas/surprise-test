from collections import defaultdict
import pandas as pd
from surprise import Reader, Dataset
from surprise import KNNWithMeans, KNNBasic, SVD
from surprise import accuracy
from surprise.model_selection import train_test_split, cross_validate

import matplotlib.pyplot as plt
import numpy as np
import json
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.metrics import precision_score



class RefinedMyAlgo():
    def __init__(self, rating_data='', data_frame='', movie_data=''):
        if rating_data:
            reader = Reader(line_format='user item rating timestamp', sep=',')
            self.ratings = Dataset.load_from_file(rating_data, reader)
#             self.trainset, self.testset = train_test_split(self.ratings, test_size=0.25)
            self.trainset = self.ratings.build_full_trainset()
            self.sim_options = {'name': 'cosine','user_based': False}
            self.df_ratings = pd.read_csv(rating_data, low_memory=False, names=['userId', 'movieId', 'rating','timestamp'])
        elif not data_frame.empty:
            reader = Reader(rating_scale=(0, 5))
            self.ratings = Dataset.load_from_df(data_frame[['userId', 'movieId', 'rating']], reader)
            self.trainset = self.ratings.build_full_trainset()
            self.sim_options = {'name': 'cosine','user_based': False}
            
        if movie_data:
            self.movies = pd.read_csv(movie_data, low_memory=False)
            self.movies['year'] = self.movies['title'].apply(lambda x: x[-5:-1])
            self.movies['title'] = self.movies['title'].apply(lambda x: x[:-7])
            self.movies['genres'] = self.movies['genres'].apply(lambda x: x.replace('|',', '))
            
    
    def random_group(self, n):
        self.users_list = list(self.df_ratings['userId'])
        random_group = random.sample(self.users_list,n)
        return random_group
        

        
    def set_k(self, k_value=''):
        if k_value:
            algo = KNNWithMeans(k=k_value, sim_options=self.sim_options)
            self.algo = algo
            self.algo.fit(self.trainset)
        else:
            algo = SVD()
            self.algo = algo
            self.algo.fit(self.trainset)
        
        
    def find_best_k(self, k_value=''):
        if k_value:
            print('K = {}'.format(k_value))
            algo = KNNWithMeans(k=k_value, sim_options=self.sim_options)
            return cross_validate(algo, self.ratings, measures=['RMSE', 'MAE'], cv=10, verbose=True)
        else:
            aux = []
            for k_value in [3, 5, 7, 10, 15, 20, 30, 40]:
                print('K = {}'.format(k_value))
                algo = KNNWithMeans(k=k_value, sim_options=self.sim_options)
                my_dict = cross_validate(algo, self.ratings, measures=['RMSE', 'MAE'], cv=10, verbose=False)
                my_dict['k_value'] = k_value
                aux.append(my_dict)
            return aux
    
    
    def set_testset(self, users):
        if users:
            user_ratings = self.trainset.ur
            movies_ids = list(self.movies['movieId'])
            global_mean=self.trainset.global_mean
            my_testset = []
            
            for user in users:
                iuid = self.trainset.to_inner_uid(str(user))
                for movie in movies_ids:
                    is_in = False
                    for rating in user_ratings[iuid]:
#                         print( 'MOVIE: {}, RATING: {}'.format(movie,bla.trainset.to_raw_iid(rating[0])) )
                        if int(movie) == int(self.trainset.to_raw_iid(int(rating[0]))):
                            is_in = True
                            break
                    if not is_in:
                        my_tuple = (str(user),str(movie),global_mean)
                        my_testset.append(my_tuple)
                        
            self.testset = my_testset
        else:
            testset = self.trainset.build_anti_testset()
            self.testset = testset
        return self.testset


    def predict_ratings(self,users=''):
        # # Predict ratings for all pairs (u, i) that are NOT in the training set.
#         testset = self.trainset.build_anti_testset()
#         self.testset = testset
        testset = self.set_testset(users)
        predictions = self.algo.test(testset)
        self.predictions = predictions
        
        
    def set_perfil_movies(self, users):
        metadata = pd.read_csv('datasets/ml-latest-small/ratings.csv', low_memory=False, names=['userId', 'movieId', 'rating','timestamp'])
        metadata = metadata.drop(columns="timestamp")

        metadata_filtered = metadata[metadata.userId.isin(users)]

        self.group_sparse_mtx = pd.pivot_table(metadata_filtered, values='rating', index=['userId'], columns=['movieId'], fill_value=0)
        
        self.perfil_movies = list(self.group_sparse_mtx)
        
    
    ### You must call self.set_perfil_movies() before
    def set_candidate_movies(self):
        candidate_movies = []
        for item in refinedMyAlgo.movies.iterrows():
        #     get the movieId of each movie in movies dataframe
            if item[1].values[0] not in self.perfil_movies:
                candidate_movies.append(item[1].values[0])
        self.candidate_movies = candidate_movies
        
        
    def calc_similarity_matrix(self):
        #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
        tfidf = TfidfVectorizer(stop_words='english')
        
        #Replace NaN with an empty string
        self.movies['title'] = self.movies['title'].fillna('')
        self.movies['genres'] = self.movies['genres'].fillna('')
        
        #Construct the required TF-IDF matrix by fitting and transforming the data
        tfidf_matrix_title = tfidf.fit_transform(self.movies['title'])
        tfidf_matrix_genres = tfidf.fit_transform(self.movies['genres'])
        
        #Compute the cosine similarity matrix
        self.cosine_sim_movies_title = cosine_similarity(tfidf_matrix_title, tfidf_matrix_title)
        self.cosine_sim_movies_genres = cosine_similarity(tfidf_matrix_genres, tfidf_matrix_genres)
        
        
    def get_similar_movies(self, references, title_weight=0.8):
        recs = []
        for movie in references:
            # Get the pairwsie similarity scores of all movies with that movie
            movie_idx = int(self.movies[self.movies['movieId']==movie['movieID']].index[0])
            sim_scores_title = list(enumerate(self.cosine_sim_movies_title[movie_idx]))
            sim_scores_genres = list(enumerate(self.cosine_sim_movies_genres[movie_idx]))
            
            # Calculate total similarity based on title and genres
            total_sim_score = []
            for i in range(len(sim_scores_title)):
#                 print("sim_score_title= {}\t sim_score_genres= {}".format(sim_scores_title[i][1], sim_scores_genres[i][1]))
                aux = (sim_scores_title[i][1]*title_weight) + (sim_scores_genres[i][1]*(1-title_weight))
                total_sim_score.append((i, aux))
#                 print("sim_score_total= {}".format(total_sim_score))
                
            # Sort the movies based on the similarity scores
            total_sim_score = sorted(total_sim_score, key=lambda x: x[1], reverse=True)
            self.total_sim_score = total_sim_score
            
            candidates_sim_score = []
            for item in total_sim_score:
                if self.movies.loc[item[0]].values[0] not in self.perfil_movies:
                    candidates_sim_score.append(item)
            
            # Get the scores of the 10 most similar movies
            candidates_sim_score = candidates_sim_score[1:11]
            
            recs.append(candidates_sim_score)
            
        return recs
    
    
    def get_relevance_score(self, recs, references):
        count = 0
        recs_dict = []
        for reference in references:
        #     print('Referência: {}\t gêneros: {}'.format(refinedMyAlgo.movies[refinedMyAlgo.movies['movieId']==reference['movieID']].values[0][1], refinedMyAlgo.movies[refinedMyAlgo.movies['movieId']==reference['movieID']].values[0][2]))

            for movie in recs[count]:
                aux = {}

                movie_id = self.movies.loc[movie[0]].values[0]
                movie_title = self.movies.loc[movie[0]].values[1]
                movie_genres = self.movies.loc[movie[0]].values[2]
                movie_similarity = movie[1]
                movie_relevance = round(((reference['rating']/5.0)+movie_similarity)/2, 3)

                aux['movie_id'] = movie_id
                aux['movie_title'] = movie_title
                aux['movie_genres'] = movie_genres
                aux['movie_similarity'] = movie_similarity
                aux['movie_relevance'] = movie_relevance

                recs_dict.append(aux)

        #         print('\tSim: {},\trelevance: {},\tmovieId: {},\ttitle: {}'.format(aux['movie_similarity'], aux['movie_relevance'], aux['movie_id'], aux['movie_title']))

            count=count+1

        recs_dict = sorted(recs_dict, key = lambda i: i['movie_relevance'],reverse=True)

        return recs_dict
    
    
    def calc_distance_item_in_list(self, item, this_list, title_weight=0.8):

        idx_i = int(self.movies[self.movies['movieId']==item['movie_id']].index[0])

        total_dist = 0
        for movie in this_list:
            
            idx_j = int(self.movies[self.movies['movieId']==int(movie['movie_id'])].index[0])

            sim_i_j = (self.cosine_sim_movies_title[idx_i][idx_j]*title_weight) + (self.cosine_sim_movies_genres[idx_i][idx_j]*(1-title_weight))
            dist_i_j = 1 - sim_i_j
            total_dist = total_dist + dist_i_j

        result = total_dist/len(this_list)
        return result
    
    
    def calc_diversity_score(self, actual_list, candidates_list, alfa=0.5):
        '''
        This function implemented here was based on MARIUS KAMINSKAS and DEREK BRIDGE paper: Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives in Recommender Systems
        func(i,R) = (relevance[i]*alfa) + (dist_i_R(i,R)*(1-alfa))
        '''
        diversity_score = []
        count = 0

        for item in candidates_list:

            aux = {}
            dist_item_R = self.calc_distance_item_in_list(item=item, this_list=actual_list)
            aux['div_score'] = (item['movie_relevance']*alfa) + (dist_item_R*(1-alfa))
            aux['idx'] = count
            diversity_score.append(aux)
            count = count + 1

        return diversity_score
    
    
    def diversify_recs_list(self, recs, n=10):
        '''
        This function implemented here was based on MARIUS KAMINSKAS and DEREK BRIDGE paper: Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives in Recommender Systems
        The Greedy Reranking Algorithm.
        '''
        diversified_list = []
        
        while len(diversified_list) < n:
            if len(diversified_list) == 0:
                diversified_list.append(recs[0])
                recs.pop(0)
            else:
                diversity_score = self.calc_diversity_score(actual_list=diversified_list, candidates_list=recs)
                diversity_score = sorted(diversity_score, key = lambda i: i['div_score'],reverse=True)
#               #  Add the item that maximize diversity in the list 
                item = diversity_score[0]
                diversified_list.append(recs[item['idx']])
#               #  Remove this item from the candidates list
                recs.pop(item['idx'])
    
        return diversified_list


    def divesify_recs_list_bounded_random(self, recs, n=10):
        '''
        This function implemented here was based on KEITH BRADLEY and BARRY SMYTH paper: Improving Recommendation Diversity
        The Bounded Random Selection Algorithm.
        '''
        diversified_list = random.sample(recs,n)

        return diversified_list

    
    def calc_dist_i_j(self, idx_i, idx_j, title_weight=0.8):
        sim_genre = self.cosine_sim_movies_genres[idx_i][idx_j]
        sim_title = self.cosine_sim_movies_title[idx_i][idx_j]
        total_sim = (sim_title*title_weight) + (sim_genre*(1-title_weight))
        dist_score = 1 - total_sim

        return dist_score
    
    
    def get_distance_matrix(self, final_recs, title_weight=0.8):
        dist_matrix = []
        for i in final_recs:
            aux = []
            movie_idx_i = int(self.movies[self.movies['movieId']==i['movie_id']].index[0])
            for j in final_recs:
                movie_idx_j = int(self.movies[self.movies['movieId']==j['movie_id']].index[0])
                dist_i_j = self.calc_dist_i_j(movie_idx_i, movie_idx_j, title_weight=0.8)
                aux.append(dist_i_j)
            dist_matrix.append(aux)
            
        return dist_matrix
    
    def get_ILD_score(self, final_recs, title_weight=0.8):
        dist_matrix = self.get_distance_matrix(final_recs, title_weight=0.8)
        np_dist_mtx = np.array(dist_matrix)
        upper_right = np.triu_indices(np_dist_mtx.shape[0], k=1)

        ild_score = np.mean(np_dist_mtx[upper_right])
        
        return ild_score
    
    
    
    # # # # # # # # # # PRECISION Module # # # # # # # # # #
    
    def get_mean(self, movie):
        converted_values = []
        for item in movie['ratings']:
            for bla in item:
                aux = float(bla)
                converted_values.append(aux)

        my_mean = sum(converted_values) / len(converted_values)
        my_mean = round(my_mean, 3)
        return my_mean
    
    
    def get_movies_means(self, movies_list, at):
        my_copy = self.df_ratings.copy()

        df_movies_ratings = my_copy.groupby('movieId')['rating'].apply(list).reset_index(name='ratings')

        movies_means = []

        for item in movies_list[:at]:
            movie = df_movies_ratings[df_movies_ratings['movieId']==item['movie_id']]
            movies_means.append(self.get_mean(movie))

        return movies_means
    
    
    def binary_mean(self, movies_mean, cutoff):
        binary_mean = []
        returned_movies = []
        for item in movies_mean:
            if item >= cutoff:
                binary_mean.append(1)
            else:
                binary_mean.append(0)

            returned_movies.append(1)

        return precision_score(binary_mean, returned_movies)
    
    
    def precision_at_offline(self, movies_list, at):
    
        global_mean = self.trainset.global_mean
        movies_list_mean = self.get_movies_means(movies_list, at)

        print("Global mean: {}, movies_list_mean: {}".format(global_mean, movies_list_mean))

        precision = self.binary_mean(movies_list_mean, global_mean)
        return precision

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def auto_run(refinedMyAlgo):
    print("\n\n-->  Initializing...")
    # refinedMyAlgo = RefinedMyAlgo(rating_data='datasets/ml-latest-small/ratings.csv', movie_data='datasets/ml-latest-small/movies.csv')
    refinedMyAlgo.set_k()






    # # # FIXED GROUP
    # my_group = [527, 387, 288, 610, 504]
    # my_group = [177, 263, 477, 274, 68]
    # my_group = [488, 226, 602, 52, 68]
    # my_group = [77, 596, 452, 243, 420]
    my_group = [448, 305, 483, 136, 66]

    # # # RANDOM GROUP
    # my_group = refinedMyAlgo.random_group(5)
    print(my_group)

    refinedMyAlgo.predict_ratings(users=my_group)
    print(len(refinedMyAlgo.predictions))





    refinedMyAlgo.set_perfil_movies(users=my_group)
    refinedMyAlgo.set_candidate_movies()

    # print(refinedMyAlgo.perfil_movies)
    # print(refinedMyAlgo.candidate_movies)
    print(refinedMyAlgo.group_sparse_mtx.head())





    print("\n\n-->  Calculating group matrix FILLED...")
    group_filled_mtx = refinedMyAlgo.group_sparse_mtx.copy()

    for index, row in group_filled_mtx.iterrows():
        for col in list(group_filled_mtx):
            if(group_filled_mtx.loc[index,col] == 0.0):
                aux = list(filter(lambda x: x.uid==str(index) and x.iid==str(col), refinedMyAlgo.predictions))
                group_filled_mtx.loc[index,col] = aux[0].est

    group_filled_mtx = group_filled_mtx.round(decimals=3)
    # group_filled_mtx.head()





    
    print("\n\n-->  Implementing least misery STRATEGY...")
    ########################################################################
    # # Implementing LEAST MISERY ending-up in a dataframe
    ########################################################################
    values = []
    labels = []
    for i in range(0,len(list(group_filled_mtx))):
        my_col = group_filled_mtx.iloc[ : ,i]
        label = my_col.name
        my_col = list(my_col)
        
        labels.append(label)
        values.append( float(min(my_col)) )
        
    # print('Array values: {}, Array labels: {}'.format(values, labels))
    agg_group_perf = pd.DataFrame(index=[900], columns=labels)

    for i in range(0,len(list(agg_group_perf))):
        agg_group_perf.iloc[0, i] = values[i]

        
    agg_group_perf = agg_group_perf.round(decimals=3)
    agg_group_perf.head()


    '''
    print("\n\n-->  Implementing MOST PLEASURE STRATEGY...")
    ########################################################################
    # # Implementing MOST PLEASURE ending-up in a dataframe
    ########################################################################
    values = []
    labels = []
    for i in range(0,len(list(group_filled_mtx))):
        my_col = group_filled_mtx.iloc[ : ,i]
        label = my_col.name
        my_col = list(my_col)
        
        labels.append(label)
        values.append( float(max(my_col)) )
        
    # print('Array values: {}, Array labels: {}'.format(values, labels))
    agg_group_perf = pd.DataFrame(index=[900], columns=labels)

    for i in range(0,len(list(agg_group_perf))):
        agg_group_perf.iloc[0, i] = values[i]
        
    agg_group_perf = agg_group_perf.round(decimals=3)
    agg_group_perf.head()
    


    print("\n\n-->  Implementing AVERAGE WITHOUT MISERY STRATEGY...")
    ########################################################################
    # # Implementing AVERAGE WITHOUT MISERY: treshold=2  ending-up in a dataframe
    ########################################################################
    values = []
    labels = []
    for i in range(0,len(list(group_filled_mtx))):
        my_col = group_filled_mtx.iloc[ : ,i]
        label = my_col.name
        my_col = list(my_col)
        
        labels.append(label)
        if float(min(my_col)) <= 2 :
            values.append( float(min(my_col)) )
        else:
            values.append( float( sum(my_col) / len(my_col) ) )
        
    # print('Array values: {}, Array labels: {}'.format(values, labels))
    agg_group_perf = pd.DataFrame(index=[900], columns=labels)

    for i in range(0,len(list(agg_group_perf))):
        agg_group_perf.iloc[0, i] = values[i]

        
    agg_group_perf = agg_group_perf.round(decimals=3)
    agg_group_perf.head()
    '''





    print("\n\n-->  Creating group preferences dict...")
    group_pref_dict = []
    for col in list(agg_group_perf):
        my_dict = {}
    #     print('Valor: {}, Coluna: {}'.format(agg_group_perf.loc[900,col], col))
        my_dict['rating'] = agg_group_perf.loc[900,col]
        my_dict['movieID'] = col
        group_pref_dict.append(my_dict)
        
    group_pref_dict = sorted(group_pref_dict, key = lambda i: i['rating'],reverse=True)
    group_pref_dict





    print("\n\n-->  Calculatin similarity matrix...")
    refinedMyAlgo.calc_similarity_matrix()






    references = group_pref_dict[0:10]
    # references = group_pref_dict

    for item in references:
        print(item)






    print("\n\n-->  Calculating recs...")
    recs = refinedMyAlgo.get_similar_movies(references)







    candidates_list = refinedMyAlgo.get_relevance_score(recs=recs, references=references)
    # print(len(candidates_list))
    print("\n\n-->  The top-20 recs are:\n")
    for item in candidates_list[0:20]:
        print('movieId: {}, relevance: {}, title:{}'.format(item['movie_id'], item['movie_relevance'], item['movie_title']))






    my_candidates = candidates_list.copy()
    final_recs_greedy = refinedMyAlgo.diversify_recs_list(recs=my_candidates)
    print("\n\n-->  The top-10 GREEDY DIVERSIFIED recs are:\n")
    for item in final_recs_greedy:
        print('movieId: {}, relevance: {}, title:{}'.format(item['movie_id'], item['movie_relevance'], item['movie_title']))

    my_candidates = candidates_list.copy()
    final_recs_random = refinedMyAlgo.divesify_recs_list_bounded_random(recs=my_candidates)
    print("\n\n-->  The top-10 RANDOM DIVERSIFIED recs are:\n")
    for item in final_recs_random:
        print('movieId: {}, relevance: {}, title:{}'.format(item['movie_id'], item['movie_relevance'], item['movie_title']))



    print('\n\n')
    print("########################################################################")
    print("#######################     EVALUATING SYSTEM    #######################")
    print("########################################################################")
    print('\n\n')


    standard_recs = candidates_list[0:10]

    ild_s = refinedMyAlgo.get_ILD_score(standard_recs, title_weight=0.8)
    ild_g = refinedMyAlgo.get_ILD_score(final_recs_greedy, title_weight=0.8)
    ild_r = refinedMyAlgo.get_ILD_score(final_recs_random, title_weight=0.8)
    p3_s = refinedMyAlgo.precision_at_offline(standard_recs, 3)
    p3_g = refinedMyAlgo.precision_at_offline(final_recs_greedy, 3)
    p3_r = refinedMyAlgo.precision_at_offline(final_recs_random, 3)
    p5_s = refinedMyAlgo.precision_at_offline(standard_recs, 5)
    p5_g = refinedMyAlgo.precision_at_offline(final_recs_greedy, 5)
    p5_r = refinedMyAlgo.precision_at_offline(final_recs_random, 5)
    p10_s = refinedMyAlgo.precision_at_offline(standard_recs, 10)
    p10_g = refinedMyAlgo.precision_at_offline(final_recs_greedy, 10)
    p10_r = refinedMyAlgo.precision_at_offline(final_recs_random, 10)

    p_3_5_10_s = [p3_s, p5_s, p10_s]  
    p_3_5_10_g = [p3_g, p5_g, p10_g]
    p_3_5_10_r = [p3_r, p5_r, p10_r]

    evaluation = dict()
    evaluation['ild_s'] = ild_s
    evaluation['ild_g'] = ild_g
    evaluation['ild_r'] = ild_r
    evaluation['p_3_5_10_s'] = p_3_5_10_s
    evaluation['p_3_5_10_g'] = p_3_5_10_g
    evaluation['p_3_5_10_r'] = p_3_5_10_r

    total_recs = dict()
    total_recs['recs_standard'] = standard_recs
    total_recs['recs_greedy'] = final_recs_greedy
    total_recs['recs_random'] = final_recs_random



    print('ILD - standard recs: {}'.format(ild_s))
    print('ILD - div greedy algo: {}'.format(ild_g))
    print('ILD - div random algo: {}'.format(ild_r))
    print('\n')
    print('P@3 - standard recs: {}\n'.format(p3_s))
    print('P@5 - standard recs: {}\n'.format(p5_s))
    print('P@10 - standard recs: {}\n'.format(p10_s))
    print('\n')
    print('\n')
    print('P@3 - div greedy algo: {}\n'.format(p3_g))
    print('P@5 - div greedy algo: {}\n'.format(p5_g))
    print('P@10 - div greedy algo: {}\n'.format(p10_g))
    print('\n')
    print('\n')
    print('P@3 - div random algo: {}'.format(p3_r))
    print('P@5 - div random algo: {}'.format(p5_r))
    print('P@10 - div random algo: {}'.format(p10_r))

    return total_recs, evaluation






for i in range(0,5):
    result = dict()
    result['recs_standard'] = []
    result['recs_greedy'] = []
    result['recs_random'] = []
    result['ild_standard'] = 0.0
    result['ild_greedy'] = 0.0
    result['ild_random'] = 0.0
    result['p_3_5_10_standard'] = []
    result['p_3_5_10_greedy'] = []
    result['p_3_5_10_random'] = []

    refinedMyAlgo = RefinedMyAlgo(rating_data='datasets/ml-latest-small/ratings.csv', movie_data='datasets/ml-latest-small/movies.csv')

    total_recs, evaluation = auto_run(refinedMyAlgo)

    result['recs_standard'] = total_recs['recs_standard']
    result['recs_greedy'] = total_recs['recs_greedy']
    result['recs_random'] = total_recs['recs_random']
    result['ild_standard'] = evaluation['ild_s']
    result['ild_greedy'] = evaluation['ild_g']
    result['ild_random'] = evaluation['ild_r']
    result['p_3_5_10_standard'] = evaluation['p_3_5_10_s']
    result['p_3_5_10_greedy'] = evaluation['p_3_5_10_g']
    result['p_3_5_10_random'] = evaluation['p_3_5_10_r']

    print('\n\n\n')
    print(' - - - - - - - -')
    print(' - - - - - - - - - -')
    print(' - - - - - - - - - - - - RUN: {}'.format(i))
    print(' - - - - - - - - - -')
    print(' - - - - - - - -')
    print('\n\n\n')


    if i is 0:
        with open("LM_G5_run.json", "w") as json_file:
            json_file.write("{}\n".format(json.dumps(result, cls=NpEncoder)))
    else:
        with open("LM_G5_run.json", "a") as json_file:
            json_file.write("{}\n".format(json.dumps(result, cls=NpEncoder)))









# result_ILD_standard = refinedMyAlgo.get_ILD_score(standard_recs, title_weight=0.8)
# print('Metric: ILD\t\tList: Standard\t\tValue: {}\n'.format(result_ILD_standard))

# result_ILD_diversified = refinedMyAlgo.get_ILD_score(final_recs, title_weight=0.8)
# print('Metric: ILD\t\tList: Diversified\t\tValue: {}\n'.format(result_ILD_diversified))

# print('Metric: Precision@10\t\tList: Standard')
# result_pat10_standard = refinedMyAlgo.precision_at_offline(standard_recs, 10)
# print(result_pat10_standard)

# print('Metric: Precision@10\t\tList: Diversified')
# result_pat10_diversified = refinedMyAlgo.precision_at_offline(final_recs, 10)
# print(result_pat10_diversified)