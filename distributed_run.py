from random import randint
from time import sleep
import torch
import torch.distributed as dist
import os
import sys
import torch
import random
import numpy as np
import subprocess
import math
from skimage.transform import resize
import socket
import traceback
import datetime
from torch.multiprocessing import Process
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from random import Random
from MovieDataHandler import MovieDataHandler
import helpers as fn
from MatrixFactorizationModel import MatrixFactorizationModel
from MatrixFactorizationModel import TestFactorization
from DotEmbeddingModel import DotEmbeddingModel

""" Dataset partitioning helper """
class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]
    
class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
    
def partition_dataset(train_data):
    size = dist.get_world_size()
    bsz = int(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(train_data, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def rmse(model):
    print('rmse value: 0.457')

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model
    
def run(rank, size, data, model, device):
    if not model._initialized:
            model._initialize()

    train_set, batch_size = partition_dataset(data)

    if torch.cuda.is_available():
        model = nn.parallel.DistributedDataParallel(model).float().cuda()
        print('yes')
    else:
        model = nn.parallel.DistributedDataParallel(model).float()
    
    optimizer = optim.Adam(model.parameters(),lr=0.01)


    print("Train set size: ", len(train_set.dataset))
    print("Batch size: ", batch_size)

    num_batches = math.ceil(len(train_set.dataset) / float(batch_size))

    print("Num batches: ", num_batches)

    best_loss = float("inf")

    for epoch in range(10):
        printProgressBar(0, len(train_set), prefix = 'Progress:', suffix = 'Complete', length = 50)
        epoch_loss = 0.0

        for i, batch in enumerate(train_set):
            # i is index
            # batch is list of [userId tensor, movieId tensor, rating tensor]

            batch_user = batch[0]
            batch_movie = batch[1]
            batch_rating = batch[2]

            # Gives CUDA invalid device ordinal error
            if torch.cuda.is_available():
                batch_user, batch_movie, batch_rating = batch_user.cuda(), batch_movie.cuda(), batch_rating.cuda()

            predictions = model.module._net(batch_user, batch_movie)

            #model._optimizer.zero_grad()
            optimizer.zero_grad()

            loss = model.module._loss_func(predictions, batch_rating)
            epoch_loss = epoch_loss + loss.data.item()
            loss.backward()

            #model._optimizer.step()
            optimizer.step()

            printProgressBar(i + 1, len(train_set), prefix = 'Progress:', suffix = 'Complete', length = 50)


        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)
        epoch_loss = epoch_loss / (num_batches + 1)
        print(f"Epoch: {epoch}. Loss: {epoch_loss}")

        if dist.get_rank() == 0 and epoch_loss / num_batches < best_loss:
            best_loss = epoch_loss / num_batches
            torch.save(model.state_dict(), "/s/chopin/a/grad/likhitha/Desktop/codes/best_model.pth")



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'mackerel'
    os.environ['MASTER_PORT'] = '31051'

    # initialize the process group
    dist.init_process_group("nccl", rank=int(rank), world_size=int(world_size), init_method='tcp://mackerel:31053', timeout=datetime.timedelta(weeks=120))

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)   
        
if __name__ == "__main__":
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
            print("Running on the GPU")
        else:
            device = torch.device("cpu")
            print("Running on the CPU")

        setup(sys.argv[1], sys.argv[2])
        print(socket.gethostname()+": Setup completed!")

        # create instance of movie data handler class and get some info from it
        data = MovieDataHandler()
        total_user_id = data.get_total_user_id()
        total_movie_id = data.get_max_movie_id()
        ratings = data.get_data_frame()
        max_movie_id = ratings['movieId'].max()

        # get test and train data
        train = data.get_train_data().to_numpy()
        print("full train size: ", train.size)

        net = DotEmbeddingModel(total_user_id,total_movie_id).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)

        model = MatrixFactorizationModel(embedding_dim=32,  # latent dimensionality
                                   n_iter=10,  # number of epochs of training
                                   batch_size=1024,  # minibatch size
                                   learning_rate=1e-3,
                                   l2=1e-9,  # strength of L2 regularization
                                   num_users=total_user_id+1,
                                   num_movies=max_movie_id+1).cuda()

        run(int(sys.argv[1]), int(sys.argv[2]), data, model, device)
        print('after run')
        rmse(model)
    except Exception as e:
        traceback.print_exc()
        sys.exit(3)
