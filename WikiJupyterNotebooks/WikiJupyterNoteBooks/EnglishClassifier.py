import pickle
import os
import shutil, sys
import gc
import findspark
findspark.init("/home/pc/TestJupyter/opt/spark-3.3.0/spark-3.3.0-bin-hadoop3")
import pyspark
from datetime import datetime
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf, col, forall

def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


def detect_lang(df, colname):
    documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
    
    languageDetector = LanguageDetectorDL.pretrained() \
        .setInputCols("document") \
        .setOutputCol("language")
    
    pipeline = Pipeline() \
        .setStages([
          documentAssembler,
          languageDetector
        ])
    
    data = df.withColumnRenamed(colname, 'text')
    result = pipeline.fit(data).transform(data)
    agg_data = result.select('text',"language.result")
    return agg_data


class EnglishClassifyier:
    seq = 1
    def __init__(self, df, colname, spark):
        self.df      = df
        self.colname = colname
        self.spark = spark
        self.engDF = None
        self.nonEngDF = None
    
    def give_lang(self):
        self.df = detect_lang(self.df, self.colname)
        
    def save_classified(self):
        self.df.show()
        is_eng = lambda x : x == 'en'
        not_eng = lambda x : x != 'en'
        self.engDF = self.df.filter(forall(col("result"), is_eng))
        self.nonEngDF = self.df.filter(forall(col("result"), not_eng))
        self.engDF.write.parquet("output/step2-English.parquet")
        self.nonEngDF.write.parquet("output/step2-nonEnglish.parquet")
        
    def refresh(self):
        self.df.write.parquet(f"temp/temp{self.seq}.parquet")
        self.df.unpersist()
        gc.collect()
        self.spark.catalog.clearCache()
        self.df = self.spark.read.parquet(f"temp/temp{self.seq}.parquet")
        self.df = self.df.repartition(18*4)
        if self.seq != 1:
            shutil.rmtree(f"temp/temp{self.seq-1}.parquet")
        self.seq = self.seq+1
    
    def execute(self):
        print_time()
        self.give_lang()
        self.refresh()
        print("assigned language")
        self.save_classified()
        print_time()