import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from tqdm.notebook import trange, tqdm
import itertools
import random

def train_test_split(data, user_col, item_col, rating_col, time_col, test_size=0.25):

    # sort data by time: test data must correspond to the period which is after the period of train data
    data = data.sort_values(by=[time_col])

    users = data[user_col].unique()
    train_size = 1 - test_size
    X_train_users, X_test_users, y_train_users, y_test_users = [], [], [], []

    # tqdm displays a progress bar
    for user in tqdm(users, desc='users'):
      # get data only for a given user
      cur_user = data[data[user_col] == user]
      # index to split data by user
      idx = int(len(cur_user) * train_size)
      X_train_users.append(cur_user[[user_col, item_col]].iloc[:idx, :].values)
      X_test_users.append(cur_user[[user_col, item_col]].iloc[idx:, :].values)
      y_train_users.append(cur_user[rating_col].values[:idx])
      y_test_users.append(cur_user[rating_col].values[idx:])

    # stack data
    X_train = pd.DataFrame(np.vstack(X_train_users), columns=[user_col, item_col])
    X_test = pd.DataFrame(np.vstack(X_test_users), columns=[user_col, item_col])
    y_train = pd.Series(np.hstack(y_train_users), name=rating_col)
    y_test = pd.Series(np.hstack(y_test_users), name=rating_col)

    return X_train, X_test, y_train, y_test

class CollaborativeFiltering(BaseEstimator):

    sim_methods = ['cosine', 'pearson']

    def __init__(self, sim_method="cosine", user_based=False):

        BaseEstimator.__init__(self)

        if sim_method not in self.sim_methods:
            raise ValueError(f"Bad value for sim_method: only {self.sim_methods} methods are supported")

        self.sim_method = sim_method
        self.user_based = user_based
        return

    def fit(self, X, y, user_col, item_col):

        X = X.copy()

        # add target as column to X
        X['y'] = y

        self.user_col = user_col
        self.item_col = item_col
        self.users = X[user_col].unique()
        self.items = X[item_col].unique()
        self.non_zero_ratings = X[[user_col, item_col]].set_index([user_col, item_col])

        if self.user_based:
            # users x items matrix
            row_dim = user_col
            col_dim = item_col
        else:
            # items x users matrix
            row_dim = item_col
            col_dim = user_col

        # calculate the average rating by user or by item
        self.mean_y = X.groupby(row_dim)['y'].mean()
        # for each user subtract from the rating their average rating
        X['y_centered'] = X['y'] - X[row_dim].apply(lambda x: self.mean_y[x])

        # create user-item matrix with (transformed) ratings
        # users by rows and items by columns
        # replace missing values with 0
        self.ratings_matrix = pd.pivot_table(X, values='y', index=row_dim, columns=col_dim, fill_value=0)
        self.ratings_matrix_centered = pd.pivot_table(X, values='y_centered', index=row_dim, columns=col_dim, fill_value=0)

        if self.user_based:
            # users x items matrix
            raw_ids_users = self.ratings_matrix.index
            raw_ids_items = self.ratings_matrix.columns
        else:
            # items x users matrix
            raw_ids_users = self.ratings_matrix.columns
            raw_ids_items = self.ratings_matrix.index

        # create dictionaries for users and items { raw id: inner id (index) in ratings }
        self.users_inner_id = dict()
        for user in self.users:
            self.users_inner_id[user] = raw_ids_users.get_loc(user)

        self.items_inner_id = dict()
        for item in self.items:
            self.items_inner_id[item] = raw_ids_items.get_loc(item)

        # convert from pd.DataFrame to numpy matrix: computing optimization
        self.ratings_matrix = self.ratings_matrix.values
        self.ratings_matrix_centered = self.ratings_matrix_centered.values

        # calculate the pairwise similarities between all samples (rows) in ratings matrix.
        if self.sim_method == 'cosine':
            self.sim_matrix = cosine_similarity(self.ratings_matrix)
        elif self.sim_method == 'pearson':
            # The Pearson correlation coefficient can be seen as a mean-centered cosine similarity,
            self.sim_matrix = cosine_similarity(self.ratings_matrix_centered)
            # self.sim_matrix = np.nan_to_num(np.corrcoef(self.ratings_matrix))

        return self

    def predict_rating(self, user, item, sim_threshhold=None):

        # return 0 for unknown users and items
        if item not in self.items or user not in self.users:
            return 0

        # cur_user_inner_id = self.users_inner_id[user]
        # cur_item_inner_id = self.items_inner_id[user]

        if self.user_based:
            # users x items matrix
            row_inner_id = self.users_inner_id[user]
            col_inner_id = self.items_inner_id[item]
            mean_rating = self.mean_y[user]
        else:
            # items x users matrix
            row_inner_id = self.items_inner_id[item]
            col_inner_id = self.users_inner_id[user]
            mean_rating = self.mean_y[item]

        sim_coefs = self.sim_matrix[row_inner_id]
        neighbours_inner_id = np.arange(len(sim_coefs))

        # don't consider the user themselves as neighbours
        neighbours_inner_id = np.delete(neighbours_inner_id, row_inner_id)
        sim_coefs = np.delete(sim_coefs, row_inner_id)

        if sim_threshhold is not None:
            neighbours_inner_id = np.where(sim_coefs > sim_threshhold)[0]
            sim_coefs = sim_coefs[neighbours_inner_id]

        # get neighbours ratings
        # neighbours_ratings = self.ratings_matrix[neighbours_inner_id, col_inner_id]
        neighbours_ratings = self.ratings_matrix_centered[neighbours_inner_id, col_inner_id]

        # calculate numerator and denominator from rating estimation formula
        numerator = sim_coefs.dot(neighbours_ratings)
        denominator = np.abs(sim_coefs).sum()

        if denominator == 0:
            prediction = mean_rating
        else:
            prediction = mean_rating + numerator / denominator

        return prediction

    def predict(self, X, sim_threshhold=None):
        # y = X[[user_col, item_col]].apply(lambda row: self.predict_rating(row[user_col], row[item_col], sim_threshhold), axis=1)
        X_np = X[[self.user_col, self.item_col]].values
        i = 0
        y = np.zeros(len(X_np))
        for row in tqdm(X_np, desc='predictions'):
            y[i] = self.predict_rating(row[0], row[1], sim_threshhold)
            i += 1
        return y

    def mean_precision_at_k(self, X_test, y_test, k, nb_random_users=None):

        if nb_random_users is None:
            selected_users = self.users.tolist()
        else:
            selected_users = random.sample(self.users.tolist(), nb_random_users)

        # all possible permutations
        X_test_all = pd.DataFrame(itertools.product(selected_users, self.items), columns=[self.user_col, self.item_col]).set_index([self.user_col, self.item_col])

        # remove permutations from self.non_zero_ratings - used in fit
        drop_permutations = X_test_all.index.intersection(self.non_zero_ratings.index)
        X_test_all = X_test_all.drop(drop_permutations).reset_index()

        y_pred_all = self.predict(X_test_all)

        col_rating = 'rating'
        y_pred_all = pd.Series(data=y_pred_all)
        data_pred = pd.concat([X_test_all[[self.user_col, self.item_col]], y_pred_all.rename(col_rating)], axis=1)
        data_test = pd.concat([X_test[[self.user_col, self.item_col]], y_test.rename(col_rating)], axis=1)

        # sort predicted ratings
        data_pred = data_pred.sort_values(col_rating, ascending=False)

        precision_by_user = {}
        for u in tqdm(selected_users, desc='users'):
            R_u_k = data_pred.loc[data_pred[self.user_col] == u, self.item_col][:k]
            L_u = data_test.loc[data_test[self.user_col] == u, self.item_col]
            if len(R_u_k) != 0:
                precision_by_user[u] = sum(np.isin(R_u_k, L_u)) / len(R_u_k)
            else:
                precision_by_user[u] = 0

        precision_by_user = pd.Series(precision_by_user)
        mean_precision = np.mean(precision_by_user)

        return mean_precision, precision_by_user

