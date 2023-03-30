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


def detect_lang(text):
    try:
        fasttext.FastText.eprint = lambda x: None # suppress warning
        i = model.predict([text])[0][0][0]
        return (i.replace("__label__",""))
    except UnboundLocalError:
        fasttext.FastText.eprint = lambda x: None
        model_file_local = SparkFiles.get('/home/pc/Assignment/PretrainedModel/lid.176.ftz')
        model = fasttext.load_model(model_file_local)
        i = model.predict([text])[0][0][0]
        return (i.replace("__label__",""))

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
        self.spark.sparkContext.addFile('/home/pc/Assignment/PretrainedModel/lid.176.ftz')
    
    def detect_language(self):
        self.df = self.df.withColumn(self.colname, col(self.colname)).withColumn("Lang", detectLang(col(self.colname)))
        
    def aggregate(self):
        self.agg = LanguageAggregor(self.df, "Lang")
        
    def save_each_aggregation(self):
        for key in self.agg.all_language:
            tempDF = self.agg.lang_df_map[key]
            tempDF.write.format('avro').save(f"output/step3-languages/{key}.avro")
    
    def refresh(self):
        self.df.write.format('avro').save(f"temp/temp{self.seq}.avro")
        self.df.unpersist()
        gc.collect()
        self.spark.catalog.clearCache()
        self.df = self.spark.read.format('avro').load(f"temp/temp{self.seq}.avro")
        self.df = self.df.repartition(18*4)
        if self.seq != 1:
            shutil.rmtree(f"temp/temp{self.seq-1}.avro")
        self.seq = self.seq+1
    
    def execute(self):
        print_time()
        self.detect_language()
        self.refresh()
        print("done detecting language")
        print_time()
        self.aggregate()
        self.save_each_aggregation()
        print("done aggregation")
        return self.agg