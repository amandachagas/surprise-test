import pandas as pd
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
from surprise import Reader, Dataset

# frame = pd.DataFrame()
# frame['person']=['me','you','him','you','him','me']
# frame['thing']=['a','a','b','c','d','d']
# frame['count']=[1,1,1,1,1,1]

# person_c = CategoricalDtype(sorted(frame.person.unique()), ordered=True)
# thing_c = CategoricalDtype(sorted(frame.thing.unique()), ordered=True)

# row = frame.person.astype(person_c).cat.codes
# col = frame.thing.astype(thing_c).cat.codes
# sparse_matrix = csr_matrix((frame["count"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))

# dfs = pd.SparseDataFrame(sparse_matrix, \
#                          index=person_c.categories, \
#                          columns=thing_c.categories, \
#                          default_fill_value=0)

# print(dfs)

metadata = pd.read_csv('ml-latest-small/ratings.csv', low_memory=False)

print(metadata.head())

print('@@@@@@@@@@@@@@@@@@@@@@@@@@@');

user = CategoricalDtype(sorted(metadata.userId.unique()), ordered=True)
movie = CategoricalDtype(sorted(metadata.movieId.unique()), ordered=True)

row = metadata.userId.astype(user).cat.codes
col = metadata.movieId.astype(movie).cat.codes
sparse_matrix = csr_matrix((metadata["rating"], (row, col)), shape=(user.categories.size, movie.categories.size))

sparse_mtx = pd.SparseDataFrame(sparse_matrix, \
                         index=user.categories, \
                         columns=movie.categories, \
                         default_fill_value=0)

# # First list is related to userId. Second list is related to movieID
print(sparse_mtx.loc[[77,596,442,243,420],[1, 110, 480]])
