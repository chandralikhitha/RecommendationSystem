# system imports
import numpy as np
import os
import torch
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from hdfs import InsecureClient

# local imports
import helpers as fn
from MovieDataHandler import MovieDataHandler
from DotEmbeddingModel import DotEmbeddingModel
from MatrixFactorizationModel import MatrixFactorizationModel

data = MovieDataHandler()

utility = data.create_utility_matrix()

#print(utility)

sparsity = data.calculate_sparsity()

print("Sparsity is: " + str(sparsity))

total_user_id = data.get_total_user_id()

total_movie_id = data.get_max_movie_id()

ratings = data.get_data_frame()

max_movie_id = ratings['movieId'].max()

# get test and train data
#user_id_train, movie_id_train, rating_train = data.get_train_data()
train = data.get_train_data()
user_id_test, movie_id_test, rating_test = data.get_test_data()

net = DotEmbeddingModel(total_user_id,total_movie_id)
optimizer = torch.optim.Adam(net.parameters(), lr = 0.1)

model = MatrixFactorizationModel(embedding_dim=128,  # latent dimensionality
                                   n_iter=10,  # number of epochs of training
                                   batch_size=1024,  # minibatch size
                                   learning_rate=1e-3,
                                   l2=1e-9,  # strength of L2 regularization
                                   num_users=total_user_id+1,
                                   num_items=max_movie_id+1)

user_ids_train_np = user_id_train.values.astype(np.int32)
movie_ids_train_np = movie_id_train.values.astype(np.int32)
ratings_train_np = rating_train.values.astype(np.float32)

model.fit(user_ids_train_np, movie_ids_train_np, ratings_train_np).cuda()


