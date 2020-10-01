import numpy as np
import os
import pandas as pd
#from hdfs import InsecureClient
from sklearn.model_selection import train_test_split
import torch.utils.data as data

#client_hdfs = InsecureClient('http://albany:30152')

class MovieDataHandler(data.Dataset):
    def __init__(self):
        ''' Init function. Reads in the movies and ratings dataset from hdfs and then
            merges them together. '''

        # with client_hdfs.read("/dataLoc/movies.csv", encoding = 'utf-8') as reader:
        # self.movies_frame = pd.read_csv("movies.csv")

        #with client_hdfs.read("/dataLoc/ratings.csv", encoding = 'utf-8') as reader:
        # self.ratings_frame = pd.read_csv("ratings.csv")

        self.data = pd.read_csv('u.data', sep='\t',
                          names=["userId", "movieId", "rating", "timestamp"])


        # split up into test and train data
        self.train, self.test = train_test_split(self.data, test_size=0.2, random_state=42)
        self.train_array = self.train.to_numpy()

    def get_train_data(self):
        #return self.train['userId'], self.train['movieId'], self.train['rating']
        return self.train

    def get_test_data(self):
         #return self.test['userId'], self.test['movieId'], self.test['rating']
        return self.test

    def __getitem__(self, index):
        # return the row of the dataframe at the index passed in
        return self.train['userId'].to_numpy()[index], self.train['movieId'].to_numpy()[index], self.train['rating'].to_numpy()[index]

    def __len__(self):
        """Return the total number of ratings in the dataset."""
        return self.train.rating.size

    def get_total_user_id(self):
        return(len(self.data['userId'].unique()))

    def get_max_movie_id(self):
        return(self.data['movieId'].max())

    def create_utility_matrix(self):
        ''' This function is used to create a utility matrix. It reads in the ratings file 
            using chunk sizes since the dataset is so large. It then goes through the chunks
            and uses the pivot_table functionality to make it in a matrix format with the 
            columns being userId and movieId, and the respective spot in the middle being the
            rating for that userId/movieId.
        '''
        utility = []

        # read ratings in with chunks since pivot table has memory issue
        with client_hdfs.read("/dataLoc/ratings.csv", encoding = 'utf-8') as reader:
            tmp_ratings = pd.read_csv(reader, chunksize=5000)

        # create pivot table out of the chunks
        for chunk in tmp_ratings:
            pivot_table = chunk.pivot_table('rating', index='userId', columns='movieId')

            utility.append(pivot_table)

        # form our similarity/utility matrix
        self.utility = pd.concat(utility, axis=0).reset_index()

        return self.utility

    def calculate_sparsity(self):
        ''' This function is used to calculate the sparsity of our data.
            This is an important step because to use collaborative filtering,
            having sparsity too low (under 0.5%) could make the results not 
            very accurate.
        '''

        # get total number of users      
        unique_users = self.ratings_frame.userId.unique().size

        # get total number of movies
        unique_movies = self.movies_frame.movieId.unique().size

        # get total number of ratings
        total_ratings = self.ratings_frame.rating.size

        # calculate sparsity
        sparsity = total_ratings / (unique_users * unique_movies)
        
        return sparsity

    def normalize_data(self):
        '''This function is used to nomralize data if necessary. '''

        # get average rating for each movie
        movie_averages = pd.DataFrame(self.data.groupby('movieId', as_index=False)['rating'].mean())

        # get the average rating each user gives
        user_averages = pd.DataFrame(self.data.groupby('userId', as_index=False)['rating'].mean())

        global_avarage = self.data['rating'].mean()

        self.data["rating"] = self.data["rating"] / self.data["rating"].max()

        return self.data

    def describe_data(self):
        print("Number of unique users: " + str(len(self.data['userId'].unique())))
        print()
        print("Number of unique movies: " + str(len(self.data['movieId'].unique())))
        print()
        print("Ratings description:")
        print(self.data['ratings'].describe())

    def get_data_frame(self):
        return self.data
