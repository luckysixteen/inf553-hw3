from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import os, sys
import time, math
import itertools
from operator import add
import csv

timeStart = time.time()

TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
CASE = int(sys.argv[3])
OUTPUT = sys.argv[4]

# Data Process
sc = SparkContext('local[*]', 'CF')
rawData = sc.textFile(TRAIN_FILE, None, False)
header = rawData.first()
rawData = rawData.filter(lambda x: x != header).map(lambda x: x.split(','))
testData = sc.textFile(TEST_FILE, None, False)
header = testData.first()
testData = testData.filter(lambda x: x != header).map(lambda x: x.split(','))

userHash = rawData.map(lambda x: (abs(hash(x[0])) % (10**9), x[0])).reduceByKey(lambda x,y: x).collect()
busiHash = rawData.map(lambda x: (abs(hash(x[1])) % (10**9), x[1])).reduceByKey(lambda x,y: x).collect()
userDict = dict()
busiDict = dict()
for user in userHash:
    userDict[user[0]] = user[1]
for busi in busiHash:
    busiDict[busi[0]] = busi[1]
userHash = testData.map(lambda x: (abs(hash(x[0])) % (10**9), x[0])).reduceByKey(lambda x,y: x).collect()
busiHash = testData.map(lambda x: (abs(hash(x[1])) % (10**9), x[1])).reduceByKey(lambda x,y: x).collect()
for user in userHash:
    if user[0] not in userDict:
        userDict[user[0]] = user[1]
for busi in busiHash:
    if busi[0] not in busiDict:
        busiDict[busi[0]] = busi[1]
ratings = rawData.map(lambda x: (
        abs(hash(x[0])) % (10**9), (
        abs(hash(x[1])) % (10**9), float(x[2]))))
test = testData.map(lambda x: (
        abs(hash(x[0])) % (10**9),
        abs(hash(x[1])) % (10**9)))

# CASE 1: Model-based CF recommendation system
if CASE == 1:
    ratings = rawData.map(lambda x: Rating(
        abs(hash(x[0])) % (10**9),
        abs(hash(x[1])) % (10**9), float(x[2])))
    rank = 10
    numIterations = 10
    model = ALS.train(ratings, rank, numIterations)
    predictions = model.predictAll(test).map(lambda r: ((r[0], r[1]), r[2]))


# CASE 2: User-based CF recommendation system
if CASE == 2:
    ratings = ratings.groupByKey().mapValues(list)
    test = test.groupByKey().mapValues(list)
    for usr in test.collect():
        i = ratings.filter(lambda x: x[0] == usr[0]).collect()[0]
        busiIni = set()
        for icount in range(len(i[1])):
            busiIni.add(i[1][icount][0])
        wList = list()
        for j in ratings.filter(lambda x: x[0] != usr[0]).collect():
            iList = list()
            jList = list()
            busiInj = set()
            for jcount in range(len(j[1])):
                busiInj.add(j[1][jcount][0])
            if len(busiIni & busiInj) > 4:
                for icount in range(len(i[1])):
                    for jcount in range(len(j[1])):
                        if i[1][icount][0] == j[1][jcount][0]:
                            iList.append(i[1][icount][1])
                            jList.append(j[1][jcount][1])
                averi = sum(iList) / len(iList)
                averj = sum(jList) / len(jList)
                number = 0
                denomi = 0
                denomj = 0
                for k in range(len(iList)):
                    a = (iList[k] - averi)
                    b = (jList[k] - averj)
                    number += (a * b)
                    denomi += (a * a)
                    denomj += (b * b)
                denom = math.sqrt(denomi) * math.sqrt(denomj)
                if denom == 0:
                    w = 0
                else:
                    w = number / denom
                wList.append((j[0], w))
        print "====>", len(wList), wList, "\n"


        # for busi in user[1]:


# TEST
# ratesAndPreds = testData.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
# RSME = ratesAndPreds.map(lambda x: (x[1][0] - x[1][1])**2).mean() ** (1/2)
# print "Root Mean Squared Error: ", RSME



# Write into CSV
predictionsPrint = list()
for pred in predictions.collect():
    res = (userDict[pred[0][0]], busiDict[pred[0][1]], pred[1])
    predictionsPrint.append(res)
    # print "predictions: ", predictionsPrint
with open(OUTPUT, 'w') as csv_output:
    csv_writer = csv.writer(csv_output)
    csv_writer.writerow(['user_id', 'business_id', 'prediction'])
    csv_writer.writerows(predictionsPrint)

timeEnd = time.time()
print "Duration: %f sec" % (timeEnd - timeStart)

# bin/spark-submit \
# --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:conf/log4j.xml" \
# --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:conf/log4j.xml" \
# ../inf553-hw3/jingyue_fu_task2.py ../inf553-hw3/input/yelp_train.csv ../inf553-hw3/input/yelp_val.csv 1 ../inf553-hw3/output/jingyue_fu_task2.csv
