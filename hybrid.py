#command to run the code
#python forth5.py data_sets data_sets\yelp_val.csv forthopt.csv
from pyspark import SparkContext, SparkConf
import pyspark
import time
import sys
import json 
import math
import pickle
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn import metrics

#timer start
start_time=time.time()

#creating a spark context
conf = pyspark.SparkConf().setMaster("local[*]").setAppName("first").setAll([('spark.executor.memory', '8g'), ('spark.executor.cores', '3'), ('spark.cores.max', '3'), ('spark.driver.memory','8g')])
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
sc = SparkContext(conf=conf)

#creating a spark context
#sc = SparkContext('local[*]','first')
sc.setLogLevel('ERROR')

#take command line inputs
input_train_folder = sys.argv[1]
input_test = sys.argv[2]
output_path = sys.argv[3]

#reading the datasets into an RDD
#take required columns from business dataset
businessRDD=sc.textFile(input_train_folder+"/business.json").map(json.loads).map(lambda dictionary: (dictionary["business_id"],float(dictionary["stars"]),float(dictionary["review_count"]),dictionary["is_open"],dictionary["latitude"],dictionary["longitude"],dictionary["state"],dictionary["city"]))#,dictionary["neighborhood"],dictionary["attributes"]
temp_business=businessRDD.collect()
business=dict()
for x in temp_business:
    business[x[0]]=x
#take required columns from user dataset
usersRDD=sc.textFile(input_train_folder+"/user.json").map(json.loads).map(lambda dictionary: (dictionary["user_id"],float(dictionary["review_count"]),dictionary["yelping_since"],float(dictionary["average_stars"]),float(dictionary["useful"])+float(dictionary["funny"])+float(dictionary["cool"]),float(dictionary["fans"]),len(dictionary["friends"].split(",")) if dictionary["friends"]!="None" else 0))
temp_users=usersRDD.collect()
users=dict()
for x in temp_users:
    users[x[0]]=x
#print(users)

#take required columns from tips dataset
tipsRDD=sc.textFile(input_train_folder+"/tip.json").map(json.loads).map(lambda dictionary: (dictionary["user_id"],dictionary["business_id"],float(dictionary["likes"])))
temp_tip=tipsRDD.collect()
tips=dict()
for x in temp_tip:
    tpl=(x[0],x[1])
    tips[tpl]=x[2]

#preparing train data for collaborative filtering
trainRDD=sc.textFile(input_train_folder+"/yelp_train.csv").repartition(18).map(lambda line: tuple(line.split(','))).distinct().filter(lambda line: True if(line[0]!='user_id') else False).persist(pyspark.StorageLevel.DISK_ONLY)

#preprocessing data for collaborative filtering
global_avg=trainRDD.map(lambda line: float(line[2])).sum()
count_pairs=trainRDD.count()
global_avg=global_avg/count_pairs
#print(global_avg)

businessUsersRDD=trainRDD.map(lambda line: (line[1],[line[0]])).reduceByKey(lambda a,b: a+b).collect()
businessUsersRDD=dict(businessUsersRDD)
#print(businessUsersRDD)

userBusinessRDD=trainRDD.map(lambda line: (line[0],[line[1]])).reduceByKey(lambda a,b: a+b).collect()
userBusinessRDD=dict(userBusinessRDD)
#print(userBusinessRDD)
number_of_users=len(userBusinessRDD)

data=trainRDD.map(lambda line: ((line[0],line[1]),float(line[2]))).collect()
data=dict(data)
#print(data)

#implementing item-item similarity based collaborative filtering
def get_weight1(business1,business2,u):
    b1=set(businessUsersRDD[business1])
    b2=set(businessUsersRDD[business2])
    if u in b1:
        b1.remove(u)
    if u in b2:
        b2.remove(u)
    common=b1.intersection(b2)
    if len(common)==0:
        return 0
    #calculating average
    summation1=0
    summation2=0
    for user in common:
        summation1=summation1+data[(user,business1)]
        summation2=summation2+data[(user,business2)]
    avg1=summation1/len(common)
    avg2=summation2/len(common)
    #calculating the pearson similarity between two businessess
    numerator=0
    denominator1=0
    denominator2=0
    for user in common:
        r1=data[(user,business1)]
        r2=data[(user,business2)]
        numerator=numerator+(r1-avg1)*(r2-avg2)
        denominator1=denominator1+pow((r1-avg1),2)
        denominator2=denominator2+pow((r2-avg2),2)
    if denominator1==0 or denominator2==0:
        return 0
    else:
        weight=numerator/(math.sqrt(denominator1)*math.sqrt(denominator2))
        return weight

def predictRating1(tpl):
    numerator=0
    denominator=0
    user=tpl[0]
    business=tpl[1]
    if user not in userBusinessRDD and business not in businessUsersRDD:
        return (user,business,global_avg)
    if user not in userBusinessRDD and business in businessUsersRDD:
        users_who_rated=businessUsersRDD[business]
        length=len(users_who_rated)
        summation=0
        for u in users_who_rated:
            summation=summation+data[(u,business)]
        return (user,business,summation/length)
    if business not in businessUsersRDD and user in userBusinessRDD:
        businessess_rated=userBusinessRDD[user]
        length=len(businessess_rated)
        summation=0
        for b in businessess_rated:
            summation=summation+data[(user,b)]
        return (user,business,summation/length)

    businesses=userBusinessRDD[user]
    nearest=list()
    for b in businesses:
        if b==business:
            continue
        weight=get_weight1(business,b,user)
        #weight=pow(weight,3)
        if weight>0.0:
            nearest.append((weight,b))
        nearest.sort(reverse=True)
    #print(nearest)
    #default voting
    if len(nearest)==0:
        return(user,business,0)

    for x in nearest:
        weight=x[0]
        b=x[1]
        numerator=numerator+data[(user,b)]*weight
        denominator=denominator+weight
    if denominator==0:
        prediction=0
    else:
        prediction=numerator/denominator
    #if prediction==0:
        #print(prediction,validationRDD[(user,business)],len(nearest))
    result=(user,business,prediction)
    return result

collab_predictions=trainRDD.map(lambda line: (line[0],line[1])).map(predictRating1).map(lambda line: ((line[0],line[1]),line[2]))
collab_predictions=dict(collab_predictions.collect())
#print(collab_predictions)
print("collaborative on train data done!")

#preparing train data for content based using xgboost
trainRDD=trainRDD.map(lambda line: (users.get(line[0],(line[0],"","","","","","")),business.get(line[1],(line[1],"","","","","","","")),tips.get((line[0],line[1]),0),float(line[2])))
train=trainRDD.collect()
train_data=list()
for x in train:
    temp=[]
    for y in x[0]:
        temp.append(y)
    for y in x[1]:
        temp.append(y)
    temp.append(x[2])
    temp.append(x[3])
    train_data.append(tuple(temp))
#print(train_data)

#preprocessing data for content based filtering 
dataframe1=pd.DataFrame(train_data)
print(dataframe1.head())

for f in dataframe1.columns:
    if dataframe1[f].dtype == 'object':
        label = preprocessing.LabelEncoder()
        label.fit(list(dataframe1[f].values))
        dataframe1[f] = label.transform(list(dataframe1[f].values))
#print(dataframe1.head())

X_train=dataframe1.drop([0,7,16],axis=1)
y_train=dataframe1[16]
#print(X_train)
#print(y_train)

X_test=dataframe1.drop([0,7,16],axis=1)
#print(X_test)

#standardising the data
# Define the scaler 
scaler = preprocessing.StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)
#print(X_train)
#print(X_test)

model1 = xgb.XGBRegressor(objective='reg:linear',learning_rate=0.1,max_depth=6)
model1.fit(X_train,y_train)
print(model1.feature_importances_)
print(model1)

#Predict for content based
output = model1.predict(data=X_test)
final_df = pd.DataFrame()
final_df["Prediction"] = output
predictions=[val for val in final_df["Prediction"]]

content_predictions=list()
count=0
for row in train_data:
    content_predictions.append(((row[0],row[7]),predictions[count]))
    count+=1
content_predictions=dict(content_predictions)
#print(content_predictions)

#iterate over all the elements of the train_data and determine the class collab or content
RecommClass=list()
for row in train_data:
    cbr=content_predictions[(row[0],row[7])]
    cor=collab_predictions[(row[0],row[7])]
    actual=row[16]
    #print(cbr,cor,actual)
    if abs(actual-cbr)<=abs(actual-cor):
        #row.append(1)
        #RecommClass.append(row)
        RecommClass.append(1)
    else:
        #row.append(0)
        #RecommClass.append(row)
        RecommClass.append(0)

#print(RecommClass)
dataframe1[17]=RecommClass
#print(dataframe1.head())
"""
trainfile=open('traindata', 'ab') 
# source, destination 
pickle.dump(dataframe1, trainfile)
trainfile.close()
"""
testRDD=sc.textFile(input_test).map(lambda line: tuple(line.split(','))).distinct().map(lambda line: (line[0],line[1])).filter(lambda line: True if(line[0]!='user_id') else False).persist(pyspark.StorageLevel.DISK_ONLY)
testRDD1=testRDD.map(lambda line: (users.get(line[0],(line[0],"","","","","","")),business.get(line[1],(line[1],"","","","","","","")),tips.get((line[0],line[1]),0)))
test=testRDD1.collect()
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

dataframe2=pd.DataFrame(test_data)

for f in dataframe2.columns:
    if dataframe2[f].dtype == 'object':
        label = preprocessing.LabelEncoder()
        label.fit(list(dataframe2[f].values))
        dataframe2[f] = label.transform(list(dataframe2[f].values))
#print(dataframe2.head())
"""
testfile=open('testdata', 'ab') 
# source, destination 
pickle.dump(dataframe2, testfile)
testfile.close()
print("Train and test dataframes are ready!!!!")
"""
#dataframe1=open('traindata', 'rb')
#dataframe1=pickle.load(dataframe1)
dataframe1=pd.DataFrame(dataframe1)
#print(dataframe1.head())

X_train=dataframe1.drop([0,7,16,17],axis=1)
y_train=dataframe1[17]
#print(X_train)
#print(y_train)

#dataframe2=open('testdata', 'rb')
#dataframe2=pickle.load(dataframe2)
dataframe2=pd.DataFrame(dataframe2)
#print(dataframe2.head())

X_test=dataframe2.drop([0,7],axis=1)
#print(X_test)

#standardising the data
# Define the scaler 
scaler = preprocessing.StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)
#print(X_train)
#print(X_test)

# Initialize the constructor
"""
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10,))) 
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
"""
#model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)
#y_pred = model.predict(X_test)
#print(y_pred)

params = {
    #'n_estimators': 100,
    'colsample_bytree': 0.8,
    #'objective': 'binary:logistic',
    'max_depth': 6,
    'min_child_weight': 1,
    'learning_rate': 0.1,
    'subsample': 0.9,
    'num_class': 2,
    'eta': 0.01,
    'gamma':1,
    'alpha':0.5,
    'lambda':1
}

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test)

model2 = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=20,
    nfold=5,
    early_stopping_rounds=5
)

# Fit
final_gb = xgb.train(params, dtrain, num_boost_round=len(model2))

preds = final_gb.predict(dtest)
final_df = pd.DataFrame()
final_df["Prediction"] = preds
y_pred=[val for val in final_df["Prediction"]]
#print(y_pred)

content_input=list()
collab_input=list()
for i in range(len(y_pred)):
    if y_pred[i]==1.0:
        content_input.append((test_data[i][0],test_data[i][7]))
    else:
        collab_input.append((test_data[i][0],test_data[i][7]))
#print(content_input)
#print(collab_input)

#Predict for content based
test_data_content=[]
for x in test_data:
    tpl=(x[0],x[7])
    if tpl in content_input:
        test_data_content.append(x)
#print(test_data_content)

dataframe3=pd.DataFrame(test_data_content)
for f in dataframe3.columns:
    if dataframe3[f].dtype == 'object':
        label = preprocessing.LabelEncoder()
        label.fit(list(dataframe3[f].values))
        dataframe3[f] = label.transform(list(dataframe3[f].values))
#print(dataframe3.head())
test=dataframe3.drop([0,7],axis=1)
test = scaler.transform(test)

output = model1.predict(data=test)
final_df = pd.DataFrame()
final_df["Prediction"] = output
predictions=[val for val in final_df["Prediction"]]

content_predictions=list()
count=0
for row in test_data_content:
    content_predictions.append(((row[0],row[7]),predictions[count]))
    count+=1
content_predictions=dict(content_predictions)
#print(content_predictions)

#predict for collab
def get_weight(business1,business2):
    b1=set(businessUsersRDD[business1])
    b2=set(businessUsersRDD[business2])
    common=b1.intersection(b2)
    if len(common)==0:
        return 0
    #calculating average
    summation1=0
    summation2=0
    for user in common:
        summation1=summation1+data[(user,business1)]
        summation2=summation2+data[(user,business2)]
    avg1=summation1/len(common)
    avg2=summation2/len(common)
    #calculating the pearson similarity between two businessess
    numerator=0
    denominator1=0
    denominator2=0
    for user in common:
        r1=data[(user,business1)]
        r2=data[(user,business2)]
        numerator=numerator+(r1-avg1)*(r2-avg2)
        denominator1=denominator1+pow((r1-avg1),2)
        denominator2=denominator2+pow((r2-avg2),2)
    if denominator1==0 or denominator2==0:
        return 0
    else:
        weight=numerator/(math.sqrt(denominator1)*math.sqrt(denominator2))
        return weight

def predictRating(tpl):
    numerator=0
    denominator=0
    user=tpl[0]
    business=tpl[1]
    if user not in userBusinessRDD and business not in businessUsersRDD:
        return (user,business,global_avg)
    if user not in userBusinessRDD and business in businessUsersRDD:
        users_who_rated=businessUsersRDD[business]
        length=len(users_who_rated)
        summation=0
        for u in users_who_rated:
            summation=summation+data[(u,business)]
        return (user,business,summation/length)
    if business not in businessUsersRDD and user in userBusinessRDD:
        businessess_rated=userBusinessRDD[user]
        length=len(businessess_rated)
        summation=0
        for b in businessess_rated:
            summation=summation+data[(user,b)]
        return (user,business,summation/length)

    businesses=userBusinessRDD[user]
    nearest=list()
    for b in businesses:
        weight=get_weight(business,b)
        #weight=pow(weight,3)
        if weight>0.0:
            nearest.append((weight,b))
        nearest.sort(reverse=True)
    #print(nearest)
    #default voting
    if len(nearest)<50:
        """
        businessess_rated=userBusinessRDD[user]
        length=len(businessess_rated)
        summation=0
        for b in businessess_rated:
            summation=summation+data[(user,b)]
        return (user,business,summation/length)
        """
        users_who_rated=businessUsersRDD[business]
        length=len(users_who_rated)
        summation=0
        for u in users_who_rated:
            summation=summation+data[(u,business)]
        return (user,business,summation/length)

    for x in nearest:
        weight=x[0]
        b=x[1]
        numerator=numerator+data[(user,b)]*weight
        denominator=denominator+weight
    if denominator==0:
        prediction=0
    else:
        prediction=numerator/denominator
    #if prediction==0:
        #print(prediction,validationRDD[(user,business)],len(nearest))
    result=(user,business,prediction)
    return result

collab_predictions=sc.parallelize(collab_input).map(predictRating).map(lambda line: ((line[0],line[1]),line[2])).collect()
collab_predictions=dict(collab_predictions)

final_output=dict()
final_output.update(content_predictions)
final_output.update(collab_predictions)
#print(final_output)
#print(len(final_output))
temp=list()
for key,val in final_output.items():
    temp.append((key[0],key[1],val)) 

#print(temp)
out_file = open(output_path,"w") 
out_file.write("user_id, business_id, prediction\n")
for x in temp[:-1]:
    out_file.write(x[0]+","+x[1]+","+str(x[2])+"\n")
out_file.write(temp[-1][0]+","+temp[-1][1]+","+str(temp[-1][2]))
out_file.close()

print("Duration: %s" % (time.time() - start_time))

validationRDD=sc.textFile(input_test).map(lambda line: tuple(line.split(','))).filter(lambda line: True if(line[0]!='user_id') else False).map(lambda line: ((line[0],line[1]),float(line[2])))
validationRDD=validationRDD.collect()
validationRDD=dict(validationRDD)
#print(len(validationRDD))

total=0
for key,val in validationRDD.items():
    a=final_output[key]
    diff=a-val
    diff=pow(diff,2)
    total=total+diff

total=total/len(final_output)
rmse = math.sqrt(total)
print("RMSE: %f" % (rmse))