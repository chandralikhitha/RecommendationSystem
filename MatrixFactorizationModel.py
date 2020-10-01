import numpy as np
import os.path as op
import pandas as pd

#import MovieDataHandler
import torch.nn as nn
import torch
import torch.optim as optim
#from hdfs import InsecureClient
# import torch.distributed as dist
import helpers as fn
from DotEmbeddingModel import DotEmbeddingModel
#import imp

class TestFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
    # create user embeddings
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                               sparse=True)
    # create item embeddings
        self.item_factors = torch.nn.Embedding(n_items, n_factors,
                                               sparse=True)

    def forward(self, user, item):
        # matrix multiplication
        return (self.user_factors(user)*self.item_factors(item)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)


class MatrixFactorizationModel(nn.Module):
    
    def __init__(self,
                 embedding_dim=128,
                 n_iter=10,
                 batch_size=256,
                 l2=1e-9,
                 learning_rate=1e-3,
                 net=None,
                 num_users=None,
                 num_movies=None, 
                 random_state=None):
        super(MatrixFactorizationModel, self).__init__()
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        
        self._num_users = num_users
        self._num_movies = num_movies
        self._net = net
        self._optimizer = None
        self._loss_func = None
        self._random_state = random_state or np.random.RandomState()
             

    #def forward(self, x):
    #    return self.net2(self.relu(self.net1(x)))
        
    def _initialize(self):
        if self._net is None:
            self._net = DotEmbeddingModel(self._num_users, self._num_movies, self._embedding_dim).cuda()
        
        self._optimizer = optim.Adam(
                self._net.parameters(),
                lr=self._learning_rate,
                weight_decay=self._l2
            )
        
        self._loss_func = fn.regression_loss_function
    
    @property
    def _initialized(self):
        return self._optimizer is not None
    
    def __repr__(self):
        return _repr_model(self)
    
    def fit(self, user_ids, item_ids, ratings, verbose=True):
        user_ids = user_ids.astype(np.int64)
        item_ids = item_ids.astype(np.int64)
        
        
        if not self._initialized:
            self._initialize()
            
        for epoch_num in range(self._n_iter):
            users, items, ratingss = fn.shuffle(user_ids,
                                            item_ids,
                                            ratings)

            user_ids_tensor = torch.from_numpy(users)
            item_ids_tensor = torch.from_numpy(items)
            ratings_tensor = torch.from_numpy(ratingss)
            epoch_loss = 0.0

            for (minibatch_num,
                 (batch_user,
                  batch_item,
                  batch_rating)) in enumerate(fn.generate_mini_batch(self._batch_size,
                                                         user_ids_tensor,
                                                         item_ids_tensor,
                                                         ratings_tensor)):

                predictions = self._net(batch_user, batch_item)

                self._optimizer.zero_grad()
                
                loss = self._loss_func(predictions,batch_rating)
                
                epoch_loss = epoch_loss + loss.data.item()
                
                loss.backward()
                self._optimizer.step()
                
            
            epoch_loss = epoch_loss / (minibatch_num + 1)

            if verbose:
                print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))
        
            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'
                                 .format(epoch_loss))
    
    
    def test(self,user_ids, item_ids, ratings):
        self._net.train(False)
        user_ids = user_ids.astype(np.int64)
        item_ids = item_ids.astype(np.int64)
        
        user_ids_tensor = torch.from_numpy(user_ids)
        item_ids_tensor = torch.from_numpy(item_ids)
        ratings_tensor = torch.from_numpy(ratings)
               
        predictions = self._net(user_ids_tensor, item_ids_tensor)
        
        loss = self._loss_func(ratings_tensor, predictions)
        return loss.data.item()
      
    def predict(self,user_ids, item_ids):
        self._net.train(False)
        user_ids = user_ids.astype(np.int64)
        item_ids = item_ids.astype(np.int64)
        
        user_ids_tensor = torch.from_numpy(user_ids)
        item_ids_tensor = torch.from_numpy(item_ids)
               
        predictions = self._net(user_ids_tensor, item_ids_tensor)
        return predictions.data
