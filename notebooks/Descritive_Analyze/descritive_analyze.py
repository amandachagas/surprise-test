from collections import defaultdict
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import json
import time


def get_movies_ratings(mini_movies, json_title):
	my_dict = []
	for index, movie in mini_movies.iterrows():
	    # access data using column names
	    print("Filme de index = {}, movieId = {}".format(index, movie['movieId']))
	    counter = 0
	    acc = 0
	    aux = 0
	    rating_list = []
	    for idx, rating in ratings.iterrows():
	        aux += 1
	        if int(rating['movieId'].item()) == movie['movieId']:
	#             print(type(int(rating['movieId'].item())))
	#             print(type(movie['movieId']))
	#             print("Só uma? {} {}".format(rating['movieId'], movie['movieId']))
	            counter += 1
	            acc += rating['rating']
	            rating_list.append(rating['rating'])
	    bla = {
	        "movieId": movie['movieId'],
	        "ratings": rating_list,
	        "counter": counter,
	        "acc": acc
	    }
	    my_dict.append(bla)

	# print("\n\n")
	# print(my_dict)

	# jname='01.json' 
	with open(json_title+'.json', 'w') as json_file:  
		json.dump(my_dict, json_file)

	elapsed_time = time.time() - start_time
	print('TIME: {}'.format(elapsed_time))


	return "-----------------------> FINALIZADA"




start_time = time.time()

movies = pd.read_csv('ml-latest-small/movies.csv', low_memory=False)
ratings = pd.read_csv('ml-latest-small/ratings.csv', low_memory=False, names=['userId', 'movieId', 'rating','timestamp'])
ratings = ratings.drop(columns=['timestamp'])

print(movies.head())
print('\n')
print(ratings.head())

get_movies_ratings(movies[0:1000],'01')
get_movies_ratings(movies[1000:2000],'02')
get_movies_ratings(movies[2000:3000],'03')
get_movies_ratings(movies[3000:4000],'04')
get_movies_ratings(movies[4000:5000],'05')
get_movies_ratings(movies[5000:6000],'06')
get_movies_ratings(movies[6000:7000],'07')
get_movies_ratings(movies[7000:8000],'08')
get_movies_ratings(movies[8000:9000],'09')
get_movies_ratings(movies[9000:],'10')



# mini_movies = movies[0:1000]

# my_dict = []
# for index, movie in mini_movies.iterrows():
#     # access data using column names
#     print("Filme de index = {}, movieId = {}".format(index, movie['movieId']))
#     counter = 0
#     acc = 0
#     aux = 0
#     rating_list = []
#     for idx, rating in ratings.iterrows():
#         aux += 1
#         if int(rating['movieId'].item()) == movie['movieId']:
# #             print(type(int(rating['movieId'].item())))
# #             print(type(movie['movieId']))
# #             print("Só uma? {} {}".format(rating['movieId'], movie['movieId']))
#             counter += 1
#             acc += rating['rating']
#             rating_list.append(rating['rating'])
#     bla = {
#         "movieId": movie['movieId'],
#         "ratings": rating_list,
#         "counter": counter,
#         "acc": acc
#     }
#     my_dict.append(bla)

# # print("\n\n")
# # print(my_dict)

# jname='01.json' 
# with open(jname, 'w') as json_file:  
# 	json.dump(my_dict, json_file)




# aux = aux_df[0:1000]

# for index, row in aux.iterrows():
#     print('Index: {}'.format(index))
#     aux.loc[index,'abstract'] = get_abstract_query_sparql(row.title)

# aux.to_csv('results.csv')
# print('\n\n============= DONE results.csv ')
# elapsed_time = time.time() - start_time
# print('TIME: {}'.format(elapsed_time))