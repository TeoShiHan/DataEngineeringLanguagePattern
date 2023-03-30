import findspark
findspark.init("/home/pc/TestJupyter/opt/spark-3.3.0/spark-3.3.0-bin-hadoop3")
import pyspark
from pyspark.sql.functions import lower, col, udf
from pyspark.sql.functions import regexp_replace
from emot.emo_unicode import UNICODE_EMOJI # For emojis
from emot.emo_unicode import EMOTICONS_EMO # For EMOTICONS
from pyspark.sql.types import StringType
from cleantext import clean
import pickle
from dateutil import parser
import os
import shutil, sys
import gc

linebreak_pattern = r"((\r\n\t)|[\n\v\t])+"
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
phone_pattern = r"((?:^|(?<=[^\w)]))(((\+?[01])|(\+\d{2}))[ .-]?)?(\(?\d{3,4}\)?/?[ .-]?)?(\d{3}[ .-]?\d{4})(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?:$|(?=\W)))|\+?\d{4,5}[ .-/]\d{6,9}"
number_pattern = r"(?:^|(?<=[^\w,.]))[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)(?:$|(?=\b))"
key_pattern = r"(\w+)(?=:):"






def print_time():
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    
def get_time():
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%H:%M:%S")

def convert_dataframe_col_to_lowercase(df, col_name):
    return df.select(lower(col(col_name)).alias(col_name))

def remove_newline_and_tabs(df, col_name):
    return df.withColumn(col_name, regexp_replace(col_name, linebreak_pattern, ''))

def remove_html_tags(df, col_name):
    ## This function will cause data loss because it also remove the non html tag
    return df.withColumn(col_name, regexp_replace(col_name, r'<[^>]+>', ''))

def standardize_links(df, col_name):
    ## This function will cause data because it also remove the non html tag
    return df.withColumn(col_name, regexp_replace(col_name, r'http\S+', '__LINK__'))

def remove_extra_whitespaces(df, col_name):
    return df.withColumn(col_name, regexp_replace(col_name, r'\s+', ' '))

def standardize_email(df, col_name):
    return df.withColumn(col_name, regexp_replace(col_name, email_pattern, '__EMAIL__'))

def standardize_tweet_alias(df, col_name):
    return df.withColumn(col_name, regexp_replace(col_name, r'@[\w]+', '__ALIAS__'))

def remove_htag(df, col_name):
    return df.withColumn(col_name, regexp_replace(col_name, "#([a-zA-Z0-9_]{1,50})", ''))

def standardize_phone_numbers(df, col_name):
    return df.withColumn(col_name, regexp_replace(col_name, phone_pattern, '__PHONE__'))

def standardize_numbers(df, col_name):
    return df.withColumn(col_name, regexp_replace(col_name, number_pattern, '__NUMBER__'))

def standardize_key_value_pair(df, col_name):
    return df.withColumn(col_name, regexp_replace(col_name, key_pattern, '__KEY__ '))

def remove_emoji_str(string):
    return clean(string, no_emoji=True, fix_unicode=False, to_ascii=False)

rem_emoji = udf(lambda z:remove_emoji_str(z),StringType())    

def remove_emoji(df, col_name):
    return df.withColumn(col_name, col(col_name)).withColumn(col_name, rem_emoji(col(col_name)))

def clear_null(df, col_name):
    return df.filter( (df[col_name] != "") | (df[col_name] != None))

class LIPP:
    seq = 1
    def __init__(self, df, col_name, spark):
        self.df = df
        self.col_name = col_name
        self.spark = spark
    
    def to_lower(self):
        self.df = convert_dataframe_col_to_lowercase(self.df, self.col_name)
        self.refresh()
    
    def no_newline(self):
        self.df = remove_newline_and_tabs(self.df, self.col_name)
        self.refresh()
    
    def no_extra_space(self):
        self.df = remove_extra_whitespaces(self.df, self.col_name)
        self.refresh()
    
    def no_html(self):
        self.df = remove_html_tags(self.df, self.col_name)
        self.refresh()

    def standardize_links(self):
        self.df = standardize_links(self.df, self.col_name)
        self.refresh()
    
    def standardize_email(self):
        self.df = standardize_email(self.df, self.col_name)
        self.refresh()
        
    def standardize_phone(self):
        self.df = standardize_phone_numbers(self.df, self.col_name)
        self.refresh()

    def standardize_tweet_alias(self):
        self.df = standardize_tweet_alias(self.df, self.col_name)
        self.refresh()
    
    def no_htag(self):
        self.df = remove_htag(self.df, self.col_name)
        self.refresh()

    def standardize_key_value_pair(self):
        self.df = standardize_key_value_pair(self.df, self.col_name)
        self.refresh()
    
    def no_emoji(self):
        self.df = remove_emoji(self.df, self.col_name)
        self.refresh()
    
    def no_null(self):
        self.df = clear_null(self.df, self.col_name)
        self.refresh()
    
    def refresh(self):
        self.df.write.format("avro").option("header", "true").save(f"temp/temp{self.seq}.avro")
        self.df.unpersist()
        gc.collect()
        self.spark.catalog.clearCache()
        self.df.checkpoint()
        self.df = self.spark.read.format('avro').load(f"temp/temp{self.seq}.avro")
        self.df = self.df.repartition(18*4)
        if self.seq != 1:
            shutil.rmtree(f"temp/temp{self.seq-1}.avro")
        self.seq = self.seq+1
        
    def get_result(self):
        print(f"process start at {get_time()}")
        self.to_lower()
        print("lowered")
        self.no_newline()
        print("no newline")
        self.no_html()
        print("no html")
        self.standardize_links()
        print("standardized links")
        self.standardize_email()
        print("standardized email")
        self.standardize_tweet_alias()
        print("standardized tweet")
        self.standardize_phone()
        print("standardized phone")
        self.standardize_key_value_pair()
        print("standardized key value pair")
        self.no_htag()
        print("no_htag")
        self.no_emoji()
        print("no emoji")
        self.no_extra_space()
        print("extra space")
        self.no_null()
        print("no null")
        print(f"process end at {get_time()}")
        return self.df