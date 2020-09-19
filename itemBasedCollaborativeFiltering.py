#command to run the code
#python second_exp.py data_sets\yelp_train.csv data_sets\yelp_val.csv secondopt.csv
#python second.py data_sets\myexample2.csv data_sets\example2_val.csv secondopt.csv

from pyspark import SparkContext, SparkConf
import pyspark
import time
import sys
import math

#timer start
start_time=time.time()

#creating a spark context
conf = pyspark.SparkConf().setMaster("local[*]").setAppName("second").setAll([('spark.executor.memory', '8g'), ('spark.executor.cores', '3'), ('spark.cores.max', '3'), ('spark.driver.memory','8g')])
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
sc = SparkContext(conf=conf)

#creating a spark context
#sc = SparkContext('local[*]','first')
sc.setLogLevel('ERROR')

#take command line inputs
input_train = sys.argv[1]
input_test = sys.argv[2]
output_path = sys.argv[3]

#reading the dataset into an RDD
originalRDD=sc.textFile(input_train).repartition(6).map(lambda line: tuple(line.split(','))).distinct().filter(lambda line: True if(line[0]!='user_id') else False).persist(pyspark.StorageLevel.DISK_ONLY)

global_avg=originalRDD.map(lambda line: float(line[2])).sum()
count_pairs=originalRDD.count()
global_avg=global_avg/count_pairs
#print(global_avg)

businessUsersRDD=originalRDD.map(lambda line: (line[1],[line[0]])).reduceByKey(lambda a,b: a+b).collect()
businessUsersRDD=dict(businessUsersRDD)
#print(businessUsersRDD)

userBusinessRDD=originalRDD.map(lambda line: (line[0],[line[1]])).reduceByKey(lambda a,b: a+b).collect()
userBusinessRDD=dict(userBusinessRDD)
#print(userBusinessRDD)
number_of_users=len(userBusinessRDD)

#applying IUF
#originalRDD=originalRDD.map(lambda line:((line[0],line[1]),float(line[2])*math.log(number_of_users/len(businessUsersRDD[line[1]]))))

data=originalRDD.map(lambda line: ((line[0],line[1]),float(line[2]))).collect()
data=dict(data)
#print(data)

#validation data remove it later
validationRDD=sc.textFile(input_test).map(lambda line: tuple(line.split(','))).filter(lambda line: True if(line[0]!='user_id') else False).distinct().map(lambda line: ((line[0],line[1]),float(line[2]))).collect()
validationRDD=dict(validationRDD)

#read the test data
testRDD=sc.textFile(input_test).repartition(6).map(lambda line: tuple(line.split(','))).distinct().map(lambda line: (line[0],line[1])).filter(lambda line: True if(line[0]!='user_id') else False)

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

predictionsRDD=testRDD.map(predictRating)
final_output=predictionsRDD.collect()

out_file = open(output_path,"w") 
out_file.write("user_id, business_id, prediction\n")
for x in final_output[:-1]:
    out_file.write(x[0]+","+x[1]+","+str(x[2])+"\n")
out_file.write(final_output[-1][0]+","+final_output[-1][1]+","+str(final_output[-1][2]))
out_file.close()

print("Duration: %s" % (time.time() - start_time))

validationRDD=sc.textFile(input_test).map(lambda line: tuple(line.split(','))).filter(lambda line: True if(line[0]!='user_id') else False).distinct().map(lambda line: ((line[0],line[1]),float(line[2]))).collect()
validationRDD=dict(validationRDD)

length=len(validationRDD)
summation=0
for x in final_output:
    temp=validationRDD[(x[0],x[1])]
    diff=x[2]-temp
    diff=pow(diff,2)
    summation=summation+diff
    #print(x[2],temp)
rmse=math.sqrt(summation/length)

print("RMSE: %f" % rmse)