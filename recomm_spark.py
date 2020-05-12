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

# load ALS model
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# initialize spark
spark = SparkSession.builder.appName('Recommendation_system').getOrCreate()
# get spark context
sc = spark.sparkContext
# sc.setCheckpointDir("checkpoints/") # this dir path must be a hdfs path if you run on cluster!

# load datasets for running on cluster
# if you are running spark-submit on cluster, you must upload these files to
# your hdfs because the default pwd would be hdfs://dumbo/user/cc6580/
# train = spark.read.load('1_perc_csv/final_train.csv', format='csv', header=True, inferSchema=True)
# val = spark.read.load('1_perc_csv/final_val.csv', format='csv', header=True, inferSchema=True)
test = spark.read.load('1_perc_csv/final_test.csv', format='csv', header=True, inferSchema=True)

# # for local running
# books = spark.read.load('book_id_map.csv', format='csv', header=True, inferSchema=True)
# train = spark.read.load('final_train.csv', format='csv', header=True, inferSchema=True)
# val = spark.read.load('final_val.csv', format='csv', header=True, inferSchema=True)
# test = spark.read.load('final_test.csv', format='csv', header=True, inferSchema=True)

# only keep the 3 columns we need
# train_data = train.select(train['user_id'], train['book_id'], train['rating'])
# val_data = val.select(val['user_id'], val['book_id'], val['rating'])
test_data = test.select(test['user_id'], test['book_id'], test['rating'])

test_data.show(3)
print(70*'*')

# create evaluator object
evaluator = RegressionEvaluator(metricName="rmse",
                                labelCol="rating",
                                predictionCol="prediction")

# load saved model
final_model = ALSModel.load("1_perc_csv/modelSaveOut")


predictions=final_model.transform(test_data)
rmse=evaluator.evaluate(predictions)
print("final test RMSE=",rmse)
predictions.show(5) # this is the test_data df with an extra prediction column
print(70*'*')
                                                                          
start_time = time.time()
recomm = final_model.recommendForAllUsers(500)
print("Showing the first 10 user recommendations")
recomm.show(10)
print ('Total Runtime for recommendation: {:.2f} seconds'.format(time.time() - start_time))
print(70*'*')

# try:# save the recommendation spark_df as a bunch of csv files to your own hdfs
#     recomm.write.mode("overwrite").format("com.intelli.spark.csv").option("header", "true").save("1_perc_csv/recommends")
#     print("recomm sparks dataframe SUCCESSFULLY saved to your hdfs under '1_perc_csv/recommends'")
#     print("this separates your dataframe into parts, to retrieve full dataframe, use -getmerge later")
#     print(70*'*')
# except:
#     print('recomm sparks dataframe could NOT be saved to your hdfs')
#     print(70*'*')
    

# stop
spark.stop()