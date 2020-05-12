#!/usr/bin/env python3

import sys
# import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random

# folder_name = 'interactions_all.csv'
# file_type = 'csv'
# seperator =','
# data = pd.concat([pd.read_csv(f, names=['user_id', 'book_id', 'is_read', 'rating', 'is_reviewed'], sep=seperator, low_memory=False) for f in glob.glob(folder_name + "/*."+file_type)],ignore_index=True)

data=pd.read_csv('goodreads/goodreads_interactions.csv', low_memory=False)

# a panda series containing only users who have read more than 10 books
more_10 = data.groupby('user_id').count()['is_read'][data.groupby('user_id').count()['is_read'] >= 10]

# only keep those users in our data
data = data[data['user_id'].isin(list(more_10.index))]
# this is a list of all unique user_id's present in the dataframe, should be the same as the length of more_10
users = data['user_id'].unique().tolist()

random.shuffle(users) # randomly shuffle the user id's
# then, all the sequential partition will be random

train_users = users[0:int(len(users) * 0.6)] # a list of user_id's for train set
val_users = users[int(len(users) * 0.6) : int(len(users) * 0.8)]# a list of user_id's for validation set
test_users = users[int(len(users) * 0.8) : ]# a list of user_id's for validation set

train = data[data['user_id'].isin(train_users)]
val = data[data['user_id'].isin(val_users)]
test = data[data['user_id'].isin(test_users)]

final_train = train

# for each USER in the validation set, move 50% of its interactions back into train, only leaving the other half
final_val = pd.DataFrame()

for user in val_users:
    user_df = val[val['user_id'] == user]
    user_train, user_val = train_test_split(user_df, train_size=0.5)
    final_train = final_train.append(user_train)
    final_val = final_val.append(user_val)
    
# for each USER in the test set, move 50% of its interactions back into train, only leaving the other half
final_test = pd.DataFrame()

for user in test_users:
    user_df = test[test['user_id'] == user]
    user_train, user_test = train_test_split(user_df, test_size=0.5)
    final_train = final_train.append(user_train)
    final_test = final_test.append(user_test)
    
final_val.to_csv('goodreads/final_val.csv', index=False)
final_train.to_csv('goodreads/final_train.csv', index=False)
final_test.to_csv('goodreads/final_test.csv', index=False)