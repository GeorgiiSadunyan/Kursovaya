import pandas as pd
import numpy as np
from sklearn import model_selection as cv
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.spatial import distance
from  sklearn.metrics.pairwise import pairwise_distances
# ar = pd.read_csv('pandas_tutorial_read.csv', delimiter=';',
#                  names = ['my_datetime', 'event', 'country', 'user_id', 'source', 'topic'])


# print(ar[ar.country=='country_2'][['user_id', 'country', 'topic']].head())


df = pd.read_csv('ratings.csv', delimiter = ',', nrows = 100000)
#print(df.head())
#print('df shape: {}'.format(df.schape))

#отберу перыые 100 000 записей
# n = 100000
# df = df[:n]
#print('df shape: {}'.format(df.shape))
# #количество строк и столбцов

n_users = len(df['userId'].unique())
n_movies = len(df['movieId'].unique())
#print(n_users, n_movies)
#702 пользователей и 8227 фильмов

# отмасштабируем идентификаторы фильмов таким образом,
# чтобы они начинались с 1 и заканчивались на n_movies
movie_ids = df['movieId'].unique()

#scaled - масштабированный
def scale_movie_id(movie_id):
    scaled = np.where(movie_ids == movie_id)[0][0] + 1
    return scaled

df['movieId'] = df['movieId'].apply(scale_movie_id)
#print(df.head())

#формирую обучающую выборку и тестовую
train_data, test_data = cv.train_test_split(df, test_size=0.2)

#print('Train shape: {}'.format(train_data.shape))
#print('Test shape: {}'.format(test_data.shape))
# Train shape: (80000, 4)
# Test shape: (20000, 4)


def rmse(prediction, ground_truth):
        # Оставим оценки, предсказанные алгоритмом, только для соотвествующего набора данных
        prediction = np.nan_to_num(prediction)[ground_truth.nonzero()].flatten()
        
        # Оставим оценки, которые реально поставил пользователь, только для соотвествующего набора данных
        ground_truth = np.nan_to_num(ground_truth)[ground_truth.nonzero()].flatten()
        
        mse = mean_squared_error(prediction, ground_truth)
        
        return sqrt(mse)


train_data_matrix = np.zeros((n_users, n_movies))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]
    
test_data_matrix = np.zeros((n_users, n_movies))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]



# считаем косинусное расстояние для пользователей и фильмов 
# (построчно и поколоночно соотвественно).

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
#user_similarity[i][j] — косинусное расстояние между i-ой строкой и j-ой строкой 
#item_similarity[i][j] — косинусное расстояние между i-ой и j-ой колонками.


print(distance.cosine([2,2],[1,5])) 
print(distance.cosine([3,3],[2,3]))
print(distance.cosine([3, 3],[1, 1.5]))
print(distance.cosine([3, 3],[1, 3]))

# 0.16794970566215628
# 0.01941932430907989
# 0.01941932430907989
# 0.10557280900008414



