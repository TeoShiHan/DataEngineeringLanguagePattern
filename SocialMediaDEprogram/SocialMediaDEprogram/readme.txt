to make sure the system can work, it is required to ensure those path passed into the arguments is
the correct one, below is the example of the argument value.


pyspark_bin_path = "/home/pc/TestJupyter/opt/spark-3.3.0/spark-3.3.0-bin-hadoop3"
python_path = "/home/pc/TestJupyter/opt/spark-3.3.0/venv-spark/bin/python39"
app_name = "social media"
data_path = "/home/pc/Assignment/SocialMedia/Main/output/sample.csv"
hdfs_working_path = "hdfs://10.123.51.78:8020/user/social_media/"
malay_stopword_path = "/home/pc/Assignment/node_modules/stopwords-ms/stopwords-ms.json"
jieba_replace_csv = "/home/pc/Assignment/SocialMedia/Main/replacement.csv"
hdfsHost = "hdfs://10.123.51.78"
hdfsPort = "8020"
hdfs_working_path_file = "/user/social_media/"
hiveWarehouseDirectory = "hdfs://10.123.51.78:8020/user/hive/warehouse"
hiveTriftServerAddress = "thrift://g2.bigtop.it:9083"

## if remove_outliers is not favourable, change to False
remove_outliers = False