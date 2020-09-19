#command to run the code
#python first1.py data_sets\yelp_train.csv firstopt3.csv
#python first1.py data_sets\myexample.csv firstopt3.csv

from pyspark import SparkContext, SparkConf
import pyspark
import time
import sys
import csv

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
input_path = sys.argv[1]
output_path = sys.argv[2]

#reading the dataset into an RDD
#here you will get user to business pairs
originalRDD=sc.textFile(input_path).repartition(18).map(lambda line: line.split(',')).filter(lambda line: True if(line[0]!='user_id') else False).persist(pyspark.StorageLevel.DISK_ONLY)

usersRDD=originalRDD.map(lambda line: line[0])
userSet=set(usersRDD.collect())
userList=list(userSet)
number_of_users=len(userList)
#print(number_of_users)

hash_functions=list()
prime_numbers=[73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 
127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 
179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 
233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 
283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 
353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 
419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 
467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 
607, 613, 617, 619, 631, 641, 643, 647, 653, 659]
#661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 
#739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 
#811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 
#877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 
#947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 
#1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 
#1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 
#1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223,
#1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 
#1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373]

number_of_hashes=100
b=50
r=2

for i in range(number_of_hashes):
    hash_functions.append((prime_numbers[i],prime_numbers[len(prime_numbers)-i-1]))
#print(hash_functions)

def get_hash_value(x):
    hashes=[number_of_users+100]*number_of_hashes
    for user in x[1]:
        for i in range(number_of_hashes):
            temp=((hash_functions[i][0]*user+hash_functions[i][1])%6607)%number_of_users
            if hashes[i]>temp:
                hashes[i]=temp
    return (x[0],hashes)

#get all users for a business: (b1:[u1,u2,.....])
#make a user-business rdd with key=business_id and values=row number of the user.
businessUsersRDD=originalRDD.map(lambda line: (line[1],[userList.index(line[0])])).reduceByKey(lambda a,b: a+b).persist(pyspark.StorageLevel.DISK_ONLY)
businessUsers=dict(businessUsersRDD.collect())

#signature matrix is the mapping of business to all the hash value so basically every column
signatureMatrixRDD=businessUsersRDD.map(get_hash_value)

def divide_into_bands(x):
    bands=list()
    for i in range(b):
        row_start=i*r
        row_end=row_start+r
        bands.append(((i,tuple(x[1][row_start:row_end])),x[0]))
    return bands

#get bands for all the businesses, in the form ((band_number,band_value),business)
bandsRDD=signatureMatrixRDD.flatMap(divide_into_bands).map(lambda line: (line[0],[line[1]])).reduceByKey(lambda a,b: a+b).filter(lambda line: True if len(line[1])>=2 else False)

def make_pairs(x):
    candidates=set()
    for i in range(len(x)-1):
        for j in range(i+1,len(x)):
            if x[i]>x[j]:
                candidates.add((x[j],x[i]))
            else:
                candidates.add((x[i],x[j]))
    return candidates

candidatesRDD=bandsRDD.map(lambda line: line[1]).flatMap(make_pairs).distinct()
"""
for x in candidatesRDD.collect():
    print(x)
"""
#finding actually similar pairs
def check_similarity(x):
    longer=1
    shorter=0
    if len(businessUsers[x[0]])>len(businessUsers[x[1]]):
        longer=0
        shorter=1
    union=set(businessUsers[x[longer]]+businessUsers[x[shorter]])
    union=len(union)
    intersection=0
    for i in range(len(businessUsers[x[longer]])):
        if businessUsers[x[longer]][i] in businessUsers[x[shorter]]:
            intersection+=1
    jaccard=intersection/union
    if jaccard>=0.5:
        return (x[0],x[1],jaccard)
    else:
        return None

actualPairsRDD=candidatesRDD.map(check_similarity).filter(lambda line: True if line!=None else False)
final_output=actualPairsRDD.collect()
final_output.sort()

"""
with open(output_path,'w', newline='') as csvfile:
    writer=csv.writer(csvfile, delimiter=' ', quotechar='|')
    writer.writerow(["user_id", "business_id", "similarity"])
    for x in final_output:
        writer.writerow(list(x))
"""

out_file = open(output_path,"w") 
out_file.write("business_id_1, business_id_2, similarity\n")
for x in final_output[:-1]:
    out_file.write(x[0]+","+x[1]+","+str(x[2])+"\n")

out_file.write(final_output[-1][0]+","+final_output[-1][1]+","+str(final_output[-1][2]))
out_file.close() 

print("Duration: %s" % (time.time() - start_time))

real=list()
with open('data_sets/pure_jaccard_similarity.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        real.append(row)

real.remove(real[0])
print(len(real))

truePositives=list()
falsePositives=list()
falseNegatives=list()

with open('firstopt3.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        if row in real:
            truePositives.append(row)
        else:
            falsePositives.append(row)

falsePositives.remove(falsePositives[0])

for x in real:
    if x not in truePositives:
        falseNegatives.append(x)
print(len(truePositives))
print(len(falsePositives))
print(len(falseNegatives))

precision=len(truePositives)/(len(truePositives)+len(falsePositives))
recall=len(truePositives)/(len(truePositives)+len(falseNegatives))

print(precision)
print(recall)