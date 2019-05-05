from pyspark import SparkContext
import os, sys
import time
from random import randint,randrange
import itertools
from operator import add
import csv

def getHashSet(HASH_NUM):
    a = [randrange(HASH_NUM) for _ in range(0, HASH_NUM)]
    b = [randrange(int(PRIME_NUM/2)) for _ in range(0, HASH_NUM)]
    return list(zip(a, b))


def getHashValue(x, hashSet):
    return (hashSet[0] * x + hashSet[1]) % PRIME_NUM


def minhash(rate):
    signiture = dict()
    hashSet = getHashSet(HASH_NUM)
    initList = []
    for _ in hashSet:
        initList.append(-1)
    for row in rate:
        for busi in row[1]:
            if busi not in signiture:
                signiture[busi] = initList[:]
    for row in rate:
        for i in range(len(hashSet)):
            hValue = getHashValue(row[0], hashSet[i])
            for busi in row[1]:
                if signiture[busi][i] == -1 or signiture[busi][i] > hValue:
                    signiture[busi][i] = hValue
    return signiture


def getBandHashValue(band):
    value = ''
    for x in band:
        value += str(x)
    return int(value) % BUCKET_PRIME


def getJaccardSim(group):
    a = set(busi[group[0]])
    b = set(busi[group[1]])
    jaccardSim = float(len(a & b)) / len(a | b)
    return jaccardSim


timeStart = time.time()

INPUT_CSV = sys.argv[1]
OUTPUT_CSV = sys.argv[2]

PRIME_NUM = 997
HASH_NUM = 120 #150
ROWS = 3
BUCKET_PRIME =  99991

# Data Process: Creat baskets
sc = SparkContext('local[*]', 'LSH')
rawData = sc.textFile(INPUT_CSV, None, False)
header = rawData.first()
rawData = rawData.filter(lambda x: x != header).map(lambda x: x.decode().split(','))
rateData = rawData.map(lambda x: (abs(hash(x[0])) % (10**9), [x[1]])).reduceByKey(lambda x,y: x+y).sortByKey()
# print rateData.take(10)

#minhash
signiture = minhash(rateData.collect())

#LSH
candidate = set()
for i in range(0, HASH_NUM, 3):
    bandDict = dict()
    bucketSet = set()
    for busi in signiture:
        bucket = getBandHashValue(signiture[busi][i: i + 3])
        if bucket not in bandDict:
            bandDict[bucket] = [busi]
        else:
            bandDict[bucket].append(busi)
            bucketSet.add(bucket)
    count = 0
    for bucket in bucketSet:
        count += 1
        for x in itertools.chain(*[itertools.combinations(bandDict[bucket], 2)]):
            simPair = tuple(sorted(x))
            candidate.add(simPair)

for i in range(0, HASH_NUM, 4):
    bandDict = dict()
    bucketSet = set()
    for busi in signiture:
        bucket = getBandHashValue(signiture[busi][i:i + 4])
        if bucket not in bandDict:
            bandDict[bucket] = [busi]
        else:
            bandDict[bucket].append(busi)
            bucketSet.add(bucket)
    count = 0
    for bucket in bucketSet:
        count += 1
        for x in itertools.chain(
                *[itertools.combinations(bandDict[bucket], 2)]):
            simPair = tuple(sorted(x))
            candidate.add(simPair)

busiData = rawData.map(lambda x: (x[1], [abs(hash(x[0])) % (10**9)])).reduceByKey(lambda x,y: x+y)
busi = dict()
for row in busiData.collect():
    busi[row[0]] = row[1]

res = list()
for group in candidate:
    jacSim = getJaccardSim(group)
    if jacSim >= 0.5:
        res.append((group[0],group[1], jacSim))
res = sorted(res, key = lambda x: x[0])
# print "Total: ", len(res)

# ==================== BRUTH FORCE ====================
# busiData = rawData.map(lambda x: (x[1], [abs(hash(x[0])) % (10**5)])).reduceByKey(lambda x,y: x+y)
# busiList = busiData.map(lambda x: x[0]).collect()
# sorted(busiList)
# busi = dict()
# for row in busiData.collect():
#     busi[row[0]] = row[1]

# res = list()
# for i in range(len(busiList)):
#     for j in range(i+1, len(busiList)):
#         a = busiList[i]
#         b = busiList[j]
#         group = (a, b)
#         jacSim = getJaccardSim(group)
#         if jacSim >= 0.5:
#             res.append((group, jacSim))
#             print (group, jacSim)

# test = ('-8O4kt8AIRhM3OUxt-pWLg', '_p64KqqRmPwGKhZ-xZwhtg')
# test = ('0Rw40S_OgNnoeGq9nQ5oGA', 'djW8gh3JJ-__NCxx1YQaHg')
# print getJaccardSim(test)
# res = list()
# for i in range(len(busi)):
#     for j in range(i+1, len(busi)):
#         a = busi[i]
#         b = busi[j]
#         group = (a, b)
#         jacSim = getJaccardSim(group)
#         if jacSim >= 0.5:
#             res.append((group, jacSim))
# print len(res)

with open(OUTPUT_CSV, 'w') as csv_output:

    csv_writer = csv.writer(csv_output)

    csv_writer.writerow(['business_id_1', 'business_id_2', 'similarity'])
    csv_writer.writerows(res)

timeEnd = time.time()
print ("Duration: %f sec" % (timeEnd - timeStart))

# bin/spark-submit \
# --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:conf/log4j.xml" \
# --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:conf/log4j.xml" \
# ../inf553-hw3/jingyue_fu_task1.py ../inf553-hw3/input/yelp_train.csv ../inf553-hw3/output/jingyue_fu_task1.csv
