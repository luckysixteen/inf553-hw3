from pyspark import SparkContext
import os, sys
import time
import itertools
from operator import add

timeStart = time.time()

THRESHOLD = int(sys.argv[1])
SUPPORT = int(sys.argv[2])
INPUT_CSV = sys.argv[3]
OUTPUT_TXT = sys.argv[4]

# Data Process: Creat baskets
sc = SparkContext('local[*]', 'SONalgorithm')
rawData = sc.textFile(INPUT_CSV, None, False)
header = rawData.first()
rawData = rawData.filter(lambda x: x != header)


timeEnd = time.time()
print "Duration: %f sec" % (timeEnd - timeStart)

# spark-submit \
# --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:conf/log4j.xml" \
# --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:conf/log4j.xml" \
# ../hw2/jingyue_fu_task2.py 70 50 ../hw2/data.csv ../hw2/jingyue_fu_task2.txt
