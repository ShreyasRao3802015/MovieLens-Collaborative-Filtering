import pandas as pd
import os

def preprocess_data(data_path):
    print("Preprocessing data...")
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(os.path.join(data_path, 'ratings.dat'), sep='::', names=r_cols, engine='python', encoding='latin-1')

    user_ids = ratings['user_id'].unique().tolist()
    user2idx = {user_id: i for i, user_id in enumerate(user_ids)}
    ratings['user_idx'] = ratings['user_id'].apply(lambda x: user2idx[x])

    movie_ids = ratings['movie_id'].unique().tolist()
    movie2idx = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    ratings['movie_idx'] = ratings['movie_id'].apply(lambda x: movie2idx[x])

    num_users = len(user2idx)
    num_items = len(movie2idx)

    print(f"Number of unique users: {num_users}")
    print(f"Number of unique items: {num_items}")
    print(f"Total ratings: {len(ratings)}")

    return ratings, num_users, num_items, user2idx, movie2idx