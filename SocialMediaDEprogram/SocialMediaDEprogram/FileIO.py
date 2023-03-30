import findspark
import os
from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame
from hdfs3 import HDFileSystem


def init_spark(
    pyspark_bin_path, 
    python_path, 
    app_name
    ):

    findspark.init(pyspark_bin_path)
    import pyspark
    from pyspark.sql import SparkSession

    os.environ["HADOOP_USER_NAME"] = "hdfs"
    os.environ["PYSPARK_PYTHON"] = python_path
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-avro_2.12:3.3.0  pyspark-shell'

    spark = SparkSession.builder \
        .master("local[*]") \
        .appName(app_name) \
        .getOrCreate()

    return spark


def read_avro(spark, path):
    return spark.read.format("avro").load(path)


def write_avro(df, path):
    return df.write.mode("overwrite").format("avro").save(path, header = 'true')


def read_csv(spark, path):
    return spark.read.csv(path)


def csv_to_avro(df, path):
    return df.write.format("avro").save(path, header = 'true')


def remove_from_hdfs(file_path):
    os.system(f"hdfs dfs -rm -R {file_path}")


def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)


def delete_path(host, port, path):
    hdfs = HDFileSystem(host=host, port=port)
    HDFileSystem.rm(path)