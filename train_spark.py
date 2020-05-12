'''
after running this code, should produce the best trained model saved under specified hdfs directory
'''

import os
import time
import sys

# data science imports
import math
import numpy as np
import pandas as pd

# spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import UserDefinedFunction, explode, desc
from pyspark.sql.types import StringType, ArrayType

# visualization imports
import seaborn as sns
import matplotlib.pyplot as plt

# load ALS model
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel


# initialize spark
spark = SparkSession.builder.appName('Recommendation_system').getOrCreate()
# get spark context
sc = spark.sparkContext
# sc.setCheckpointDir("checkpoints/") # this dir path must be a hdfs path if you run on cluster!

# load datasets for running on cluster
# if you are running spark-submit on cluster, you must upload these files to
# your hdfs because the default pwd would be hdfs://dumbo/user/cc6580/
train = spark.read.load('1_perc_csv/final_train.csv', format='csv', header=True, inferSchema=True)
val = spark.read.load('1_perc_csv/final_val.csv', format='csv', header=True, inferSchema=True)
test = spark.read.load('1_perc_csv/final_test.csv', format='csv', header=True, inferSchema=True)

# # for local running
# books = spark.read.load('book_id_map.csv', format='csv', header=True, inferSchema=True)
# train = spark.read.load('final_train.csv', format='csv', header=True, inferSchema=True)
# val = spark.read.load('final_val.csv', format='csv', header=True, inferSchema=True)
# test = spark.read.load('final_test.csv', format='csv', header=True, inferSchema=True)

# only keep the 3 columns we need
train_data = train.select(train['user_id'], train['book_id'], train['rating'])
val_data = val.select(val['user_id'], val['book_id'], val['rating'])
test_data = test.select(test['user_id'], test['book_id'], test['rating'])

train_data.show(3)
print(70*'*')

# create evaluator object
evaluator = RegressionEvaluator(metricName="rmse",
                                labelCol="rating",
                                predictionCol="prediction")


def get_best_als(train_data, val_data, maxIter = 10, regs = [0.001, 0.01, 0.05, 0.1, 0.2],
                 ranks = [8, 10, 12, 14, 16, 18, 20]):
    """
    grid search function to select the best model based on RMSE of
    validation data
    Parameters
    ----------
    train_data: spark DF with columns ['user_id', 'book_id', 'rating']
    
    val_data: spark DF with columns ['user_id', 'book_id', 'rating']
    
    maxIter: int, max number of learning iterations
    
    regs: list of float, one dimension of hyper-param tuning grid
    
    ranks: list of int, one dimension of hyper-param tuning grid
    
    Return
    ------
    The best fitted ALS model with lowest RMSE score on validation data
    """
    # initialize variables
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    evaluator = RegressionEvaluator(metricName="rmse",
                                    labelCol="rating",
                                    predictionCol="prediction")
    for rank in ranks:
        for reg in regs:
            als=ALS(maxIter=maxIter,regParam=reg,rank=rank,
                    userCol="user_id",
                    itemCol="book_id",
                    ratingCol="rating",
                    coldStartStrategy="drop", # if this is not set, rmse will return nan
                    nonnegative=True)
            model = als.fit(train_data)
            predictions = model.transform(val_data)
            rmse = evaluator.evaluate(predictions)
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_regularization = reg
                best_model = model
                print('{} latent factors and regularization = {}: '
                  'validation RMSE is {}'.format(rank, reg, rmse))
    
    print('\nThe best model has {} latent factors and '
          'regularization = {}'.format(best_rank, best_regularization))
    
    # save the best model as output under the current path for later access
    # this saves the model as a directory of meta data files
    try:
        best_model.save("1_perc_csv/modelSaveOut") # this saves your model in your HDFS home directory
        print("model saved under '1_perc_csv/' as 'modelSaveOut'")
    except:
        best_model.write().overwrite().save("1_perc_csv/modelSaveOut")
        print("model already saved under '1_perc_csv/' as 'modelSaveOut', overwrite it")
    return best_model, best_rank, best_regularization

# grid search and select best model
start_time = time.time()
final_model, best_rank, best_reg = get_best_als(train_data, val_data)
print ('Total Runtime for tuning: {:.2f} seconds'.format(time.time() - start_time))
print(70*'*')


def plot_learning_curve(train_data, val_data, reg, ranking, arr_iters = list(range(1,11)) ):
    """
    Plot function to show learning curve of ALS over different maxIters
    Parameters
    ----------
    train_data: spark DF with columns ['user_id', 'book_id', 'rating']
    
    val_data: spark DF with columns ['user_id', 'book_id', 'rating']
    
    regs: a float for regParam value
    
    ranks: a int for rank value
    
    arr_iters = list of int for maxIter value, defaults to [1,2,3,...,11]
    
    Return
    ------
    
    """
    evaluator = RegressionEvaluator(metricName="rmse",
                                    labelCol="rating",
                                    predictionCol="prediction")
    errors = []
    for maxIter in arr_iters:
        # train ALS model
        als=ALS(maxIter=maxIter, regParam=reg, rank=ranking,
                userCol="user_id",
                itemCol="book_id",
                ratingCol="rating",
                coldStartStrategy="drop", # if this is not set, rmse returns nan
                    nonnegative=True)
        model = als.fit(train_data)
        predictions = model.transform(val_data)
        rmse = evaluator.evaluate(predictions)
        errors.append(rmse)
        
#     # plot
#     # plt.figure(figsize=(12, 6))
#     plt.plot(arr_iters, errors)
#     plt.xlabel('number of iterations')
#     plt.ylabel('RMSE')
#     plt.title('ALS Learning Curve')
#     # plt.grid(True)
#     plt.show()
    
    return errors
                                                                          
                                                                          
                                                                          
# only generate learning curve if you run locally
# pyspark doesn't have any plotting functionality (yet). If you want to plot something, you can bring the data out of the Spark Context and into your "local" Python session, where you can deal with it using any of Python's many plotting libraries.

# we will make a table instead
start_time = time.time()
learning_errors = plot_learning_curve(train_data, val_data, reg=best_reg, ranking=best_rank, arr_iters=list(range(1,11)))
learning_table = pd.DataFrame({'maxIter':list(range(1,11)), 'rmse':learning_errors})
print(learning_table.to_string(index=False)) # show table for learning curve

print ('Total Runtime for learning curve: {:.2f} seconds'.format(time.time() - start_time))
print(70*'*')



# if you want to directly load the best model from output instead of waiting for training result
final_model = ALSModel.load("1_perc_csv/modelSaveOut")


predictions=final_model.transform(test_data)
rmse=evaluator.evaluate(predictions)
print("final test RMSE=",rmse)
predictions.show(5) # this is the test_data df with an extra prediction column
print(70*'*')
                                                                          
# recomm = final_model.recommendForAllUsers(500)
# recomm.show(5)
# print(70*'*')

# try:# save the recommendation spark_df as a bunch of csv files to your own hdfs
#     recomm.write.csv("1_perc_csv/recommends" ,header=True)
#     print("recomm sparks dataframe SUCCESSFULLY saved to your hdfs under '1_perc_csv/recommends'")
#     print("this separates your dataframe into parts, to retrieve full dataframe, use -getmerge later")
#     print(70*'*')
# except:
#     print('recomm sparks dataframe could NOT be saved to your hdfs')
#     print(70*'*')
    

# stop
spark.stop()