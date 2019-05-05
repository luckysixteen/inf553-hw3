from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import os, sys
import time, math
from random import randint, randrange
import itertools
from operator import add
import csv

def get_pearson_correlation(vector_a, vector_b):
    '''
    Args:
        vector_a: a list of values in first vector
        vector_b: a list of values in second vector
    Return:
        A float number 
    '''
    average_a = sum(vector_a) / len(vector_a)
    average_b = sum(vector_b) / len(vector_b)
    number = 0
    denominator_a = 0
    denominator_b = 0
    for i in range(len(vector_a)):
        a = (vector_a[i] - average_a)
        b = (vector_b[i] - average_b)
        number += (a * b)
        denominator_a += (a * a)
        denominator_b += (b * b)
    denominator = math.sqrt(denominator_a) * math.sqrt(denominator_b)
    if denominator == 0:
        return 1.0
    else:
        return float(number/denominator) + 1

timeStart = time.time()

TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
CASE = int(sys.argv[3])
OUTPUT = sys.argv[4]

# Data Process
sc = SparkContext('local[*]', 'CF')
rawData = sc.textFile(TRAIN_FILE, None, False)
header = rawData.first()
rawData = rawData.filter(lambda x: x != header).map(lambda x: x.decode().split(','))
testData = sc.textFile(TEST_FILE, None, False)
header = testData.first()
testData = testData.filter(lambda x: x != header).map(lambda x: x.decode().split(','))


# CASE 1: Model-based CF recommendation system
if CASE == 1:
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
    ratings = rawData.map(lambda x: Rating(
        abs(hash(x[0])) % (10**9),
        abs(hash(x[1])) % (10**9), float(x[2])))
    rank = 4
    numIterations = 15
    regularization_parameter = 0.1
    model = ALS.train(ratings, rank, numIterations, regularization_parameter, seed=0)
    predictions = model.predictAll(test).map(lambda r: ((r[0], r[1]), r[2]))

    # TEST
    ratesAndPreds = testData.map(lambda r: ((abs(hash(r[0])) % (10**9), abs(hash(r[1])) % (10**9)), float(r[2]))).join(predictions)
    RSME = ratesAndPreds.map(lambda x: (x[1][0] - x[1][1])**2).mean() **(0.5)
    print ("Root Mean Squared Error: ", RSME)

    # Write into CSV
    predictionsPrint = list()
    for pred in predictions.collect():
        res = (userDict[pred[0][0]], busiDict[pred[0][1]], pred[1])
        predictionsPrint.append(res)


# CASE 2: User-based CF recommendation system
if CASE == 2:
    ratings = rawData.map(lambda x: (x[0], (x[1], float(x[2]))))
    raters = rawData.map(lambda x: (x[1], (x[0], float(x[2]))))
    test = testData.map(lambda x: (x[0], x[1]))

    ratings = ratings.groupByKey().mapValues(list).collectAsMap()
    raters = raters.map(lambda x: (x[0], x[1][0])).groupByKey().mapValues(
        list).collectAsMap()
    users_business_set = dict()
    users_average_rate = dict()
    for key in ratings:
        busi = set()
        total = 0
        for count in range(len(ratings[key])):
            total += ratings[key][count][1]
            busi.add(ratings[key][count][0])
        ratings[key] = dict(ratings[key])
        users_average_rate[key] = total / len(busi)
        users_business_set[key] = busi
    test = test.groupByKey().mapValues(list).collect()
    pred = list()

    for usr in test:
        user = usr[0]
        usr_aver = users_average_rate[usr[0]]
        usr_business = users_business_set[usr[0]]

        for busi in usr[1]:
            no_data = ((usr[0], busi), usr_aver)
            # Step1: Check if the business to be rated is in the record.
            if busi in raters:
                candidates = raters[busi]
            else:
                pred.append(no_data)
                continue

            # Step2: Check if the user rating the business has common rating with user to be predicted.
            common_business = list()
            for neighbor in candidates:
                common = usr_business & users_business_set[neighbor]
                if len(common) > 0:
                    common_business.append((neighbor, list(common)))
            if len(common_business) == 0:
                pred.append(no_data)
                continue

            # Step3: Compute pearson correlation coefficient between target user and the neighbor.
            weight_list = list()
            for com in common_business:
                neighbor = com[0]
                usr_rating_list = list()
                neighbor_rating_list = list()
                for b in com[1]:
                    usr_rating_list.append(ratings[user][b])
                    neighbor_rating_list.append(ratings[neighbor][b])

                weight = get_pearson_correlation(usr_rating_list,
                                                neighbor_rating_list)
                weight_list.append((neighbor, weight))

            # Step4: Compute a prediction from a weighted combination of the selected neighbors’ ratings.
            number = 0
            deno = 0
            for w in weight_list:
                number += (ratings[w[0]][busi] - users_average_rate[w[0]]) * w[1]
                deno += abs(w[1])
            if deno == 0:
                pred.append(no_data)
            else:
                predValue = usr_aver + (number / deno)
                pred.append(((usr[0], busi), predValue))
                # print ((usr[0], busi), predValue)

    predictions = sc.parallelize(pred)

    # TEST
    ratesAndPreds = testData.map(lambda r: ((r[0], r[1]), float(r[2]))).join(predictions)
    RSME = ratesAndPreds.map(lambda x: (x[1][0] - x[1][1])**2).mean() **(0.5)
    print ("Root Mean Squared Error: ", RSME)

    # Write into CSV
    predictionsPrint = list()
    for pred in predictions.collect():
        res = (pred[0][0], pred[0][1], pred[1])
        predictionsPrint.append(res)


# CASE 3: Item-based CF recommendation system
if CASE == 3:
    ratings = rawData.map(lambda x: (x[0], (x[1], float(x[2]))))
    raters  = rawData.map(lambda x: (x[1], (x[0], float(x[2]))))
    test = testData.map(lambda x: (x[0],x[1]))

    raters = raters.groupByKey().mapValues(list).collectAsMap()
    ratings = ratings.map(lambda x: (x[0], x[1][0])).groupByKey().mapValues(list).collectAsMap()
    business_user_set = dict()
    business_average_rate = dict()
    for key in raters:
        user = set()
        total = 0
        for i in range(len(raters[key])):
            total += raters[key][i][1]
            user.add(raters[key][i][0])
        raters[key] = dict(raters[key])
        business_average_rate[key] = total / len(user)
        business_user_set[key] = user
    test = test.groupByKey().mapValues(list).collect()
    pred = list()

    for usr in test:
        user = usr[0]
        # Step1: Check if the user and the business have rating record.
        if user in ratings:
            candidates = ratings[user]
            total = 0
            for b in ratings[user]:
                total += raters[b][user]
            busi_average = total / len(candidates)
            for busi in usr[1]:
                no_data = ((user, busi), busi_average)
                if busi in raters:
                    # Step2: Check if the rating business of the user has common users with business to be predicted.
                    common_user = list()
                    for neighbor in candidates:
                        common = business_user_set[neighbor] & business_user_set[busi]
                        if len(common) > 0:
                            common_user.append((neighbor,list(common)))
                    if len(common_user) == 0:
                        pred.append(no_data)
                        continue

                    # Step3: Compute pearson correlation coefficient between two businesses.
                    weight_list = list()
                    for com in common_user:
                        neighbor = com[0]
                        business_rating_list = list()
                        neighbor_rating_list = list()
                        for u in com[1]:
                            business_rating_list.append(raters[busi][u])
                            neighbor_rating_list.append(raters[neighbor][u])
                        weight = get_pearson_correlation(business_rating_list, neighbor_rating_list)
                        weight_list.append((raters[neighbor][user], weight))

                    # Step4: Summation over neighborhood set N of items rated by user.
                    number = 0
                    deno = 0
                    for w in weight_list:
                        number += (w[0] * w[1])
                        deno += abs(w[1])
                    if deno == 0:
                        pred.append(no_data)
                    else:
                        pred_value = number / deno
                        pred.append(((user, busi), pred_value))
                else:
                    pred.append(no_data)
        else:
            for busi in usr[1]:
                total = 0
                no_data = ((user, busi), business_average_rate[busi])
                pred.append(no_data)
    predictions = sc.parallelize(pred)

    # TEST
    ratesAndPreds = testData.map(lambda r: ((r[0], r[1]), float(r[2]))).join(predictions)
    RSME = ratesAndPreds.map(lambda x: (x[1][0] - x[1][1])**2).mean() **(0.5)
    print ("Root Mean Squared Error: ", RSME)

    # Write into CSV
    predictionsPrint = list()
    for pred in predictions.collect():
        res = (pred[0][0], pred[0][1], pred[1])
        predictionsPrint.append(res)

# CASE 4: Item-based CF recommendation system with Jaccard based LSH
if CASE == 4:
    # jacacardS = time.time()
    # rate_data = rawData.map(lambda x: (abs(hash(x[0])) % (10**9), [x[1]])).reduceByKey(lambda x,y: x+y).sortByKey()
    # signiture = minhash(rate_data.collect())
    # busi_data = rawData.map(lambda x: (x[1], x[0])).reduceByKey(lambda x,y: x+y)
    # business_user_set = dict()
    # for row in busi_data.collect():
    #     business_user_set[row[0]] = row[1]
    # jacacardE = time.time()
    # print("LSH time: %f sec" % (jacacardE - jacacardS))

    ratings = rawData.map(lambda x: (x[0], (x[1], float(x[2]))))
    raters = rawData.map(lambda x: (x[1], (x[0], float(x[2]))))
    test = testData.map(lambda x: (x[0], x[1]))

    raters = raters.groupByKey().mapValues(list).collectAsMap()
    ratings = ratings.map(lambda x: (x[0], x[1][0])).groupByKey().mapValues(list).collectAsMap()
    business_user_set = dict()
    business_average_rate = dict()
    for key in raters:
        user = set()
        total = 0
        for i in range(len(raters[key])):
            total += raters[key][i][1]
            user.add(raters[key][i][0])
        raters[key] = dict(raters[key])
        business_average_rate[key] = total / len(user)
        business_user_set[key] = user
    test = test.groupByKey().mapValues(list).collect()
    pred = list()

    for usr in test:
        userS = time.time()
        user = usr[0]
        # Step1: Check if the user and the business have rating record.
        if user in ratings:
            candidates = ratings[user]
            total = 0
            for b in ratings[user]:
                total += raters[b][user]
            busi_average = total / len(candidates)
            for busi in usr[1]:
                no_data = ((user, busi), busi_average)
                if busi in raters:
                    # Step2: Check if the rating business of the user has common users with business to be predicted.
                    # common_user = set()
                    # for i in range(0, HASH_NUM, 3):
                    #     goal = getBandHashValue(signiture[busi][i:i + 3])
                    #     for neighbor in candidates:
                    #         if getBandHashValue(signiture[neighbor][i:i + 3]) == goal:
                    #             common_user.add(neighbor)

                    # for neighbor in candidates:
                    #     common = business_user_set[
                    #         neighbor] & business_user_set[busi]
                    #     if len(common) > 0:
                    #         common_user.append((neighbor, list(common)))
                    # if len(common_user) == 0:
                    #     pred.append(no_data)
                    #     continue

                    # Step3: Compute Jaccard similarty between two businesses.
                    weight_list = list()
                    for neighbor in candidates:
                        # a = set(business_user_set[neighbor])
                        # b = set(business_user_set[busi])
                        common = business_user_set[neighbor] & business_user_set[busi]
                        union = business_user_set[neighbor] | business_user_set[busi]
                        if len(common) > 0:
                            jaccardSim = float(len(common)) / len(union)
                            if jaccardSim >= 0.6:
                                weight_list.append((raters[neighbor][user], jaccardSim))
                    if len(weight_list) == 0:
                        pred.append(no_data)
                        continue
                    # for com in list(common_user):
                    #     a = set(business_user_set[com])
                    #     b = set(business_user_set[busi])
                    #     jaccardSim = float(len(a & b)) / len(a | b)
                    #     weight_list.append((raters[com][user], jaccardSim))

                    # Step4: Summation over neighborhood set N of items rated by user.
                    number = 0
                    deno = 0
                    for w in weight_list:
                        number += (w[0] * w[1])
                        deno += abs(w[1])
                    if deno == 0:
                        pred.append(no_data)
                    else:
                        pred_value = number / deno
                        pred.append(((user, busi), pred_value))
                else:
                    pred.append(no_data)
        else:
            for busi in usr[1]:
                total = 0
                no_data = ((user, busi), business_average_rate[busi])
                pred.append(no_data)
        userE = time.time()
        # print("user time: %f sec" % (userE - userS))
    predictions = sc.parallelize(pred)

    # TEST
    ratesAndPreds = testData.map(lambda r: ((r[0], r[1]), float(r[2]))).join(
        predictions)
    RSME = ratesAndPreds.map(lambda x: (x[1][0] - x[1][1])**2).mean()**(0.5)
    print("Root Mean Squared Error: ", RSME)
    print ("Explaination: Using LSH in item-based CF system not much effect on speed, but setting 0.5 as threshold of jaccard similarty improve the accuracy of the system!")

    # Write into CSV
    predictionsPrint = list()
    for pred in predictions.collect():
        res = (pred[0][0], pred[0][1], pred[1])
        predictionsPrint.append(res)


# print "predictions: ", predictionsPrint
with open(OUTPUT, 'w') as csv_output:
    csv_writer = csv.writer(csv_output)
    csv_writer.writerow(['user_id', 'business_id', 'prediction'])
    csv_writer.writerows(predictionsPrint)

timeEnd = time.time()
print ("Duration: %f sec" % (timeEnd - timeStart))

# bin/spark-submit \
# --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:conf/log4j.xml" \
# --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:conf/log4j.xml" \
# ../inf553-hw3/jingyue_fu_task2.py ../inf553-hw3/input/yelp_train.csv ../inf553-hw3/input/yelp_val.csv 1 ../inf553-hw3/output/jingyue_fu_task2.csv
