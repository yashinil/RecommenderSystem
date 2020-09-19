#command to run the code
#python third2.py data_sets data_sets\yelp_val.csv thirdoutput.csv
from pyspark import SparkContext, SparkConf
import pyspark
import time
import sys
import json 
import math
import xgboost as xgb
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,precision_score, recall_score, accuracy_score

#creating a spark context
conf = pyspark.SparkConf().setMaster("local[*]").setAppName("first").setAll([('spark.executor.memory', '8g'), ('spark.executor.cores', '3'), ('spark.cores.max', '3'), ('spark.driver.memory','8g')])
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
sc = SparkContext(conf=conf)

#creating a spark context
#sc = SparkContext('local[*]','first')
sc.setLogLevel('ERROR')

#timer start
start_time=time.time()

#take command line inputs
input_train_folder = sys.argv[1]
input_test = sys.argv[2]
output_path = sys.argv[3]

#reading the dataset into an RDD
businessRDD=sc.textFile(input_train_folder+"/business.json").map(json.loads).map(lambda dictionary: (dictionary["business_id"],dictionary["city"],dictionary["state"],dictionary["stars"],dictionary["review_count"],dictionary["is_open"]))#,dictionary["attributes"]
temp_business=businessRDD.collect()
business=dict()
for x in temp_business:
    business[x[0]]=x[1:]

usersRDD=sc.textFile(input_train_folder+"/user.json").map(json.loads).map(lambda dictionary: (dictionary["user_id"],dictionary["review_count"],dictionary["yelping_since"],dictionary["average_stars"]))
temp_users=usersRDD.collect()
users=dict()
for x in temp_users:
    users[x[0]]=x[1:]

trainRDD=sc.textFile(input_train_folder+"/yelp_train.csv").map(lambda line: tuple(line.split(','))).filter(lambda line: True if(line[0]!='user_id') else False).map(lambda line: (users[line[0]],business[line[1]],float(line[2])))
train=trainRDD.collect()
train_data=list()
for x in train:
    temp=[]
    for y in x[0]:
        temp.append(y)
    for y in x[1]:
        temp.append(y)
    temp.append(x[2])
    train_data.append(tuple(temp))
#print(train_data)

testRDD=sc.textFile(input_test).map(lambda line: tuple(line.split(','))).filter(lambda line: True if(line[0]!='user_id') else False).map(lambda line: (users[line[0]],business[line[1]],float(line[2])))
test=testRDD.collect()
test_data=list()
for x in test:
    temp=[]
    for y in x[0]:
        temp.append(y)
    for y in x[1]:
        temp.append(y)
    temp.append(x[2])
    test_data.append(tuple(temp))
#print(test_data)

dataframe1=pd.DataFrame(train_data)
#print(dataframe1.head())
dataframe2=pd.DataFrame(test_data)
#print(dataframe2.head())

for f in dataframe1.columns:
    if dataframe1[f].dtype == 'object':
        label = preprocessing.LabelEncoder()
        label.fit(list(dataframe1[f].values))
        dataframe1[f] = label.transform(list(dataframe1[f].values))
#print(dataframe1.head())

x_train=dataframe1.drop([8],axis=1)
y_train=dataframe1[8]
#print(x_train)
#print(y_train)

for f in dataframe2.columns:
    if dataframe2[f].dtype == 'object':
        label = preprocessing.LabelEncoder()
        label.fit(list(dataframe2[f].values))
        dataframe2[f] = label.transform(list(dataframe2[f].values))
#print(dataframe2.head())

x_test=dataframe2.drop([8],axis=1)
y_test=dataframe2[8]
y_test=[val for val in y_test]
#print(x_test)
#print(y_test)

#Fitting XGB regressor 
params = {
    #'n_estimators': 100,
    'colsample_bytree': 0.8,
    #'objective': 'binary:logistic',
    'max_depth': 8,
    'min_child_weight': 1,
    'learning_rate': 0.1,
    'subsample': 0.9,
    'num_class': 16,
    'eta': 0.01,
    'gamma':1,
    'alpha':0.5,
    'lambda':1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

model = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=20,
    nfold=5,
    early_stopping_rounds=5
)

# Fit
final_gb = xgb.train(params, dtrain, num_boost_round=len(model))

preds = final_gb.predict(dtest)
final_df = pd.DataFrame()
final_df["Prediction"] = preds
predictions=[val for val in final_df["Prediction"]]
#print(predictions)

rmse = math.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse))

print("Duration: %s" % (time.time() - start_time))