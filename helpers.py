import numpy as np
import torch.nn as nn

class ItemEmbedding(nn.Embedding):
    """ This class is used as an embedding layer that initialises
        its values to a normal variable which is scaled by the inverse
        of the emedding dimension."""

    def reset_parameters(self):
        """ Initialize parameters."""

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class BiasEmbedding(nn.Embedding):
    """ This class is used as an embedding layer for biases."""

    def reset_parameters(self):
        """ Initialize parameters."""

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


def shuffle(*arrays):
    """ This function is used to shuffle around the indices and 
        arrays so that it is not in any specific order. """

    random_state = np.random.RandomState()
    shuffle_indices = np.arange(len(arrays[0]))
    random_state.shuffle(shuffle_indices)

    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)


def generate_mini_batch(batch_size, *tensors):
    """ This function is used to generate a mini batch
        of size "bath_size" which is passed in. """

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def regression_loss_function(predicted_ratings, observed_ratings):
    return ((observed_ratings - predicted_ratings) ** 2).mean()