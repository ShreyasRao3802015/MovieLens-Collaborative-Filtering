import torch
import torch.nn as nn



class CollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_items, d_model, bias):
        '''
        n_users : number of users
        n_items : number of items
        d_model : embedding dimensions of user, item embeddings
        bias : whether bias needs to be added to user, item embeddings
        '''
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.d_model = d_model
        self.bias = bias

        self.user_embedding = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.d_model)
        self.item_embedding = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.d_model)
        if self.bias:
            self.user_bias = nn.Embedding(num_embeddings=self.n_users, embedding_dim=1)
            self.item_bias = nn.Embedding(num_embeddings=self.n_items, embedding_dim=1)

    def forward(self, user_idxs, item_idxs):
        