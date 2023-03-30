from nltk.tokenize import sent_tokenize
import findspark
findspark.init("/home/pc/TestJupyter/opt/spark-3.3.0/spark-3.3.0-bin-hadoop3")
import pyspark
from pyspark.sql.functions import lower, col, udf
from pyspark.sql.functions import regexp_replace
from emot.emo_unicode import UNICODE_EMOJI # For emojis
from emot.emo_unicode import EMOTICONS_EMO # For EMOTICONS
from pyspark.sql.types import StringType, ArrayType
from cleantext import clean
import pickle
from dateutil import parser
import os
import shutil, sys
import gc

def get_spark():
    import findspark
    findspark.init("/home/pc/TestJupyter/opt/spark-3.3.0/spark-3.3.0-bin-hadoop3")
    import pyspark
    import random
    import os
    from pyspark.sql import SparkSession

    os.environ["PYSPARK_PYTHON"]="/home/pc/TestJupyter/opt/spark-3.3.0/venv-spark/bin/python39"
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-avro_2.12:3.3.0  pyspark-shell'

    spark = SparkSession.builder \
        .master("local[*]")\
        .appName("hive hbase")\
        .config("hive.metastore.uris", "thrift://g2.bigtop.it:9083")\
        .enableHiveSupport()\
        .getOrCreate()
    
    spark.sparkContext.setCheckpointDir('/home/pc/Assignment/SocialMedia/M1 - word frequency/Main/checkpoints')
    
    return spark


spark = get_spark()


from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame
def union_all(*dfs):
    return reduce(DataFrame.unionAll, dfs)


def read_avro(path):
    return spark.read.format("avro").load(path)

def write_avro(df, path):
    return df.write.format("avro").save(path)


def get_sentenct_token(t):
    return sent_tokenize(t, "english")

engSentence = udf(lambda t:get_sentenct_token(t),ArrayType(StringType()))    
def get_end_senDF(df, coln):
    return df.withColumn(coln, engSentence(col(coln)))


