from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from LangAgg import *
from pyspark import SparkFiles
from LangAgg import *
import fasttext
from dateutil import parser
import os
import shutil, sys
import gc
from pyspark import SparkFiles
from LangAgg import *
import fasttext

def detect_lang(text):
    model_file_local = SparkFiles.get('/home/pc/Assignment/PretrainedModel/lid.176.bin')
    model = fasttext.load_model(model_file_local)
    return model.predict([text])[0][0][0]


detectLang = udf(lambda z: detect_lang(z))


def print_time():
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


class GeneralLangClassifier:
    seq = 1
    
    def __init__(self, df, colname, spark):
        self.df = df
        self.colname = colname
        self.spark = spark
        self.agg = None
        self.spark.sparkContext.addFile('/home/pc/Assignment/PretrainedModel/lid.176.bin')
    
    def detect_language(self):
        self.df = self.df.withColumn(self.colname, col(self.colname)).withColumn("Lang", detectLang(col(self.colname)))
        
    def aggregate(self):
        self.agg = LanguageAggregor(self.df, "Lang")
        
    
    def execute(self):
        print_time()
        print("start detecting language")
        self.detect_language()
        print("done detecting language")
        print_time()
        return self.df