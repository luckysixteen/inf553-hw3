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
raters = rawData.map(lambda x: (
        abs(hash(x[1])) % (10**9), (
        abs(hash(x[0])) % (10**9), float(x[2]))))
test = testData.map(lambda x: (
        abs(hash(x[0])) % (10**9),
        abs(hash(x[1])) % (10**9)))

# CASE 1: Model-based CF recommendation system
if CASE == 1:
    ratings = rawData.map(lambda x: Rating(
        abs(hash(x[0])) % (10**9),
        abs(hash(x[1])) % (10**9), float(x[2])))
    rank = 1
    numIterations = 15
    model = ALS.train(ratings, rank, numIterations)
    predictions = model.predictAll(test).map(lambda r: ((r[0], r[1]), r[2]))


# CASE 2: User-based CF recommendation system
if CASE == 2:
    ratings = ratings.groupByKey().mapValues(list).collectAsMap()
    raters = raters.map(lambda x: (x[0], x[1][0])).groupByKey().mapValues(list).collectAsMap()
    busiSetDict = dict()
    averDict = dict()
    for key in ratings:
        busi = set()
        total = 0
        for count in range(len(ratings[key])):
            total += ratings[key][count][1]
            busi.add(ratings[key][count][0])
        ratings[key] = dict(ratings[key])
        averDict[key] = total / len(busi)
        busiSetDict[key] = busi
    test = test.groupByKey().mapValues(list)
    pred = list()

    for usr in test.collect():
        i = ratings[usr[0]]
        usrAver = averDict[usr[0]]
        busiIni = busiSetDict[usr[0]]

        for busi in usr[1]:
            noData = ((usr[0], busi), usrAver)
            busiStart = time.time()
            if busi in raters:
                candidates = raters[busi]
            else:
                pred.append(noData)
                continue
            commonBusi = list()

            for j in candidates:
                common = busiIni & busiSetDict[j]
                if len(common) > 0:
                    commonBusi.append((j,list(common)))
            if len(commonBusi) == 0:
                pred.append(noData)
                continue
            # if len(commonBusi) >20:
            #     commonBusi = sorted(commonBusi, key = lambda x: len(x[1]), reverse = True)
            #     commonBusi = commonBusi[:20]

            wList = list()
            jstart = time.time()
            for com in commonBusi:
                j = com[0]
                iList = list()
                jList = list()
                for b in com[1]:
                    iList.append(i[b])
                    jList.append(ratings[j][b])
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
                wList.append((j, w, averj))

            jend = time.time()
            # print "--busi time: %f sec" % (jstart - busiStart)
            # print "     j time: %f sec" % (jend - jstart)

            member = 0
            deno = 0
            for w in wList:
                member += (ratings[w[0]][busi] - w[2]) * w[1]
                deno += abs(w[1])
            if deno == 0:
                pred.append(noData)
            else:
                predValue = usrAver + (number / deno)
                pred.append(((usr[0], busi), predValue))
                # print ((usr[0], busi), predValue)

    predictions = sc.parallelize(pred)

# TEST
ratesAndPreds = testData.map(lambda r: ((abs(hash(r[0])) % (10**9), abs(hash(r[1])) % (10**9)), float(r[2]))).join(predictions)
RSME = ratesAndPreds.map(lambda x: (x[1][0] - x[1][1])**2).mean() **(0.5)
print "Root Mean Squared Error: ", RSME



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
