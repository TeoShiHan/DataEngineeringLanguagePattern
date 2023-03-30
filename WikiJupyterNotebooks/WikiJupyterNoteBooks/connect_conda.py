import findspark
findspark.init("/home/pc/TestJupyter/opt/spark-3.3.0/spark-3.3.0-bin-hadoop3")
import pyspark
import random
import os
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"]="/home/pc/anaconda3/envs/conda_env/bin/python3.7"
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-avro_2.12:3.3.0  pyspark-shell'

spark = SparkSession.builder \
	.master("local[*]")\
    .appName("hive hbase")\
    .config("hive.metastore.uris", "thrift://g2.bigtop.it:9083")\
    .enableHiveSupport()\
    .getOrCreate()