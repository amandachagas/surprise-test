from collections import defaultdict
import pandas as pd
import numpy as np
import json

movie_data = 'datasets/ml-latest-small/movies.csv'
#  names=['movieId', 'movieId', 'rating','timestamp']
df_movies = pd.read_csv(movie_data, low_memory=False)
df_movies['year'] = df_movies['title'].apply(lambda x: x[-5:-1])
df_movies['title'] = df_movies['title'].apply(lambda x: x[:-7])
df_movies['genres'] = df_movies['genres'].apply(lambda x: x.replace('|',', '))

tags_data = 'datasets/ml-latest-small/tags.csv'
df_tags = pd.read_csv(tags_data, low_memory=False)
print(df_tags.head(10))


my_dict = []

my_df = pd.DataFrame(columns = ['movieId','tags'])

for index, row in df_movies.iterrows():
    aux = []
    print(row.movieId)
    for index_sub, row_sub in df_tags.iterrows():
        if row_sub.movieId == row.movieId:
            aux.append(row_sub.tag)
            
    if aux:
        my_dict.append(aux)
        my_df = my_df.append(
            {
                'movieId': row.movieId,
                'tags': aux
            },
            ignore_index=True
        )

# print(my_dict)
print(my_df.head())

my_df.to_csv('datasets/grouped_tags.csv', index=False)