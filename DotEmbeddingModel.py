import torch.nn as nn
import helpers as fn

class DotEmbeddingModel(nn.Module):
    """ This class is used to define the forward function 
        used for our overall project. """

    def __init__(self,
                 num_users,
                 num_movies,
                 embedding_dim=32):
        
        super(DotEmbeddingModel, self).__init__()
        
        # save the embedding dimension 
        self.embedding_dim = embedding_dim
        
        # use the custom item embedding model to get our user embeddings and movie embeddings
        self.user_embeddings = fn.ItemEmbedding(num_users, embedding_dim)
        self.movie_embeddings = fn.ItemEmbedding(num_movies, embedding_dim)

        # use the custom bias embedding model to get our user and movie biases
        self.user_biases = fn.BiasEmbedding(num_users, 1)
        self.movies_biases = fn.BiasEmbedding(num_movies, 1)
                
        
    def forward(self, user_ids, movie_ids):
        
        user_embedding = self.user_embeddings(user_ids)
        movie_embeddings = self.movie_embeddings(movie_ids)

        user_embedding = user_embedding.squeeze()
        movie_embeddings = movie_embeddings.squeeze()

        self.user_ids = user_ids
        self.movie_ids = movie_ids


        user_bias = self.user_biases(user_ids).squeeze()
        movie_bias = self.movies_biases(movie_ids).squeeze()

        # get the dot from the equation below
        dot = (user_embedding * movie_embeddings).sum(1)

        return dot + user_bias + movie_bias