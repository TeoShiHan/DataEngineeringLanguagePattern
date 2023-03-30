import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql.functions import regexp_replace
from dateutil import parser
from pyspark.sql.functions import col, udf, size, split
from pyspark.sql.types import StringType, ArrayType
from pyspark import SparkFiles
import contractions
import numpy as np
import regex as re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import jieba
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
import jieba
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, ArrayType
from pyspark import SparkFiles
from pyspark.serializers import PickleSerializer, AutoBatchedSerializer
from nltk.tokenize import sent_tokenize
import nltk
from nltk.corpus import stopwords
import json
import unidecode


linebreak_pattern = r"((\r\n\t)|[\n\v\t])+"
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
phone_pattern = r"((?:^|(?<=[^\w)]))(((\+?[01])|(\+\d{2}))[ .-]?)?(\(?\d{3,4}\)?/?[ .-]?)?(\d{3}[ .-]?\d{4})(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?:$|(?=\W)))|\+?\d{4,5}[ .-/]\d{6,9}"
number_pattern = r"(?:^|(?<=[^\w,.]))[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)(?:$|(?=\b))"
key_pattern = r"(\w+)(?=:):"
num = "[0-9]"

punct_reg = """☺|☻|♥|♦|♣|♠|•|◘|○|◙|♂|♀|♪|♫|☼|►|◄|↕|‼|¶|§|▬|↨|↑|↓|→|←|∟|↔|▲|▼|#|%|&|,|-|:|;|<|=|>|@|]|_|`|¢|£|¥|₧|ƒ|ª|º|¿|⌐|¬|½|¼|¡|«|»|░|▒|▓|│|┤|╡|╢|╖|╕|╣|║|╗|╝|╜|╛|┐|└|┴|┬|├|─|┼|╞|╟|╚|╔|╩|╦|╠|═|╬|╧|╨|╤|╥|╙|╘|╒|╓|╫|╪|┘|┌|█|▄|▌|▐|▀|α|ß|Γ|π|Σ|σ|µ|τ|Φ|Ω|δ|∞|φ|ε|∩|≡|±|≥|≤|⌠|⌡|÷|≈|°|∙|·|√|ⁿ|²|■|\\~|\\.|\\?|\\*|\\!|\\✓|\\^|？|｡|。|＂|＃|＄|％|＆|＇|（）|＊|＋|，|－|／|：|；|＜|＝|＞|＠|［|＼|］|＾|＿|｀|｛|｜|｝|～|｟｠|｢|｣､|、|〃|》|「|」|『|』|【|】|〔|〕|〖〗|〘|〙|〚|〛|〜|〝|〞|〟|〰|〾|〿|–|—|‘|’|‛|“|”|„|‟|…|‧|﹏|\\!|\\#|\\$|\\%|\\^|\\&|\\*|\\(|\\)|\\-|\\"|\\'
"""

regex = '☺|☻|♥|♦|♣|♠|•|◘|○|◙|♂|♀|♪|♫|☼|►|◄|↕|‼|¶|§|▬|↨|↑|↓|→|←|∟|↔|▲|▼|#|%|&|,|-|:|;|<|=|>|@|]|_|`|¢|£|¥|₧|ƒ|ª|º|¿|⌐|¬|½|¼|¡|«|»|░|▒|▓|│|┤|╡|╢|╖|╕|╣|║|╗|╝|╜|╛|┐|└|┴|┬|├|─|┼|╞|╟|╚|╔|╩|╦|╠|═|╬|╧|╨|╤|╥|╙|╘|╒|╓|╫|╪|┘|┌|█|▄|▌|▐|▀|α|ß|Γ|π|Σ|σ|µ|τ|Φ|Ω|δ|∞|φ|ε|∩|≡|±|≥|≤|⌠|⌡|÷|≈|°|∙|·|√|ⁿ|²|■|\\~|\\.|\\?|\\*|\\!|\\✓|\\^|？|｡|。|＂|＃|＄|％|＆|＇|（）|＊|＋|，|－|／|：|；|＜|＝|＞|＠|［|＼|］|＾|＿|｀|｛|｜|｝|～|｟｠|｢|｣､|、|〃|》|「|」|『|』|【|】|〔|〕|〖〗|〘|〙|〚|〛|〜|〝|〞|〟|〰|〾|〿|–|—|‘|’|‛|“|”|„|‟|…|‧﹏|\\!|\\#|\\$|\\%|\\^|\\&|\\*|\\(|\\)|\\-|\\"'


number_pattern = r"(?:^|(?<=[^\w,.]))[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)(?:$|(?=\b))"


malayStopWords = set(json.load(open("/home/pc/Assignment/node_modules/stopwords-ms/stopwords-ms.json")))
stopwordsSet = set(stopwords.words('english') + stopwords.words('chinese') ).union(malayStopWords)


def rem_accent(text):
    return unidecode.unidecode(text)

udfRemacc = udf(lambda x : rem_accent(x), StringType())



def rem_stop(text):
    token = jieba.cut(text, cut_all=True)
    l = []
    for x in token:
        if x in stopwordsSet or x == "" or x == " ":
            continue
        else:
            l.append(x)
    return l


R_sw = udf(lambda x : rem_stop(x), ArrayType(StringType()))


def get_no_stop_word_df(df, colname):
    return df.withColumn(colname, R_sw(col(colname)))


def remove_number_spaces_punctuations_in_array(arr):
    i = [re.sub(punct_reg, '', x) for x in arr]
    ii = [re.sub(number_pattern, 'NUMBER', x) for x in i]
    iii = [x for x in ii if (x != "") and (x != " ")]
    return iii


rem_num_space_null = udf(lambda x : remove_number_spaces_punctuations_in_array(x), ArrayType(StringType()))

def get_non_punc_df_and_std_num(df, colname):
     return df.withColumn(colname, rem_num_space_null(col(colname)))


def remove_punct_symbol(df, colname):
    return df.withColumn(colname, regexp_replace(colname, regex, ' '))

def is_valid_date(date_str):
    try:
        parser.parse(date_str)
        return True
    except:
        return False

def convert_date_time_to_tag_ENG(text):
        return ' '.join([w if not is_valid_date(w) else "DATETIME" for w in text.split()])

standardize_date_ENG = udf(lambda x : convert_date_time_to_tag_ENG(x), StringType())
    
def get_std_date_df_ENG(df, colname):
        return df.withColumn(colname, col(colname)).withColumn(colname, standardize_date_ENG(col(colname)))

    
def convert_date_time_to_tag_BC(text):
        return ([w if not is_valid_date(w) else "DATETIME" for w in tokenize_chinese(text)])
    
standardize_date_BC = udf(lambda x : convert_date_time_to_tag_BC(x), ArrayType(StringType()))

def get_std_date_df_BC(df, colname):
        return df.withColumn(colname, col(colname)).withColumn(colname, standardize_date_BC(col(colname)))
    
    
    
def remove_punct_symbol2(df, colname):
    return df.withColumn(colname, regexp_replace(colname, regex2, ' '))

def remove_contractions(text):
    return ' '.join([contractions.fix(word) for word in text.split()])

noContr = udf(lambda x : remove_contractions(x), StringType())

def remove_contr(df, colname):
    return df.withColumn(colname, col(colname)).withColumn(colname, noContr(col(colname)))

def get_sentence_df_ENG_BM(df):
    documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")
    sentence = SentenceDetector() \
        .setInputCols(["document"]) \
        .setOutputCol("sentences")
    sentenceDL = SentenceDetectorDLModel \
        .pretrained("sentence_detector_dl", "en") \
        .setInputCols(["document"]) \
        .setOutputCol("sentencesDL")
    pipeline = Pipeline().setStages([
        documentAssembler,
        sentence,
        sentenceDL
    ])
    
    return pipeline.fit(df).transform(df).selectExpr(f"explode(sentencesDL.result) as text")

from pyspark.sql.functions import sentences, lit
def get_sentence(df, colname):
    return df.select(sentences(df[colname], lit("en"), lit("uk")))


def tokenize_chinese(text):
    l = list(jieba.cut(text))
    return [ i for i in l if i != " " and not isinstance(i, int)]

def count_token(arr):
    return len(arr)


tokenize_chineseDF = udf(lambda x : tokenize_chinese(x), ArrayType(StringType()))



def chinese_para_to_sentence(text):
    if text != "":
        return re.split('(。|！|\!|\.|？|\?)', text)
    else:
        return []

getChiS = udf(lambda x : chinese_para_to_sentence(x), ArrayType(StringType()))


def get_df_chiSen(df, colname):
    return df.withColumn(colname, getChiS(col(colname)))



def get_df_token(df, colname):
    return df.withColumn(colname, col(colname)).withColumn("token", tokenize_chineseDF(col(colname))).withColumn("len", size(col("token")))

def split_fullstop(x):
    return x.split('.')

split_fs = udf(lambda x : split_fullstop(x), ArrayType(StringType()))

def split_from_fullstop(df, colname):
    return df.withColumn(colname, col(colname)).withColumn("sentences", split_fs(col(colname)))


from matplotlib import pyplot as plt
import numpy as np
def draw_pie(cake, count): 
    fig = plt.figure(figsize =(10, 7))
    plt.pie(count, labels = cake)
    plt.show()
    
    
def get_sentenct_token(t):
    return sent_tokenize(t, "english")

engSentence = udf(lambda t:get_sentenct_token(t),ArrayType(StringType()))

def get_end_senDF(df, coln):
    return df.withColumn(coln, engSentence(col(coln)))




# import malaya
# from malaya.tokenizer import SentenceTokenizer

def get_BM_sentence(text):
    s_tokenizer = malaya.tokenizer.SentenceTokenizer()
    return s_tokenizer.tokenize(text)

    
bmSentence = udf(lambda t:get_BM_sentence(t),ArrayType(StringType()))

def get_bm_senDF(df, coln):
    return df.withColumn(coln, bmSentence(col(coln)))


def preprocessing(df, col_name):
    return  \
    df.withColumn(col_name, lower(col(col_name)))\
    .withColumn(col_name, regexp_replace(col_name, linebreak_pattern, '')) \
    .withColumn(col_name, regexp_replace(col_name, r'<[^>]+>', '')) \
    .withColumn(col_name, regexp_replace(col_name, r'http\S+', '')) \
    .withColumn(col_name, regexp_replace(col_name, email_pattern, 'EMAIL')) \
    .withColumn(col_name, regexp_replace(col_name, r'@[\w]+', 'ALIAS'))\
    .withColumn(col_name, regexp_replace(col_name, phone_pattern, 'PHONE'))\
    .withColumn(col_name, regexp_replace(col_name, key_pattern, '')) \
    .withColumn(col_name, regexp_replace(col_name, "#([a-zA-Z0-9_]{1,50})",''))\
    .withColumn(col_name, regexp_replace(col_name, "#([a-zA-Z0-9_]{1,50})",''))\
    .withColumn(col_name, regexp_replace(col_name, r'\s+', ' ')) \
    .withColumn(col_name, regexp_replace(col_name, num, '')) \
    .filter( (df[col_name] != "") | (df[col_name] != None))



def _to_java_object_rdd(rdd):  
    """ Return a JavaRDD of Object by unpickling
    It will convert each Python object into Java object by Pyrolite, whenever the
    RDD is serialized in batch or not.
    """
    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return rdd.ctx._jvm.org.apache.spark.mllib.api.python.SerDe.pythonToJava(rdd._jrdd, True)


def convert_size_bytes(size_bytes):
    """
    Converts a size in bytes to a human readable string using SI units.
    """
    import math
    import sys

    if not isinstance(size_bytes, int):
        size_bytes = sys.getsizeof(size_bytes)

    if size_bytes == 0:
        return "0B"

    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def get_df_size(df, spark):
    JavaObj = _to_java_object_rdd(df.rdd)
    nbytes = spark.sparkContext._jvm.org.apache.spark.util.SizeEstimator.estimate(JavaObj)
    return convert_size_bytes(nbytes)


import pyspark.sql.functions as f


def find_outliers(df):

    # Identifying the numerical columns in a spark dataframe
    numeric_columns = [column[0] for column in df.dtypes if column[1]=='int']

    # Using the `for` loop to create new columns by identifying the outliers for each feature
    for column in numeric_columns:

        less_Q1 = 'less_Q1_{}'.format(column)
        more_Q3 = 'more_Q3_{}'.format(column)
        Q1 = 'Q1_{}'.format(column)
        Q3 = 'Q3_{}'.format(column)

        # Q1 : First Quartile ., Q3 : Third Quartile
        Q1 = df.approxQuantile(column,[0.25],relativeError=0)
        Q3 = df.approxQuantile(column,[0.75],relativeError=0)
        
        # IQR : Inter Quantile Range
        # We need to define the index [0], as Q1 & Q3 are a set of lists., to perform a mathematical operation
        # Q1 & Q3 are defined seperately so as to have a clear indication on First Quantile & 3rd Quantile
        IQR = Q3[0] - Q1[0]
        
        #selecting the data, with -1.5*IQR to + 1.5*IQR., where param = 1.5 default value
        less_Q1 =  Q1[0] - 1.5*IQR
        more_Q3 =  Q3[0] + 1.5*IQR
        
        isOutlierCol = 'is_outlier_{}'.format(column)
        
        df = df.withColumn(isOutlierCol,f.when((df[column] > more_Q3) | (df[column] < less_Q1), 1).otherwise(0))
    

    # Selecting the specific columns which we have added above, to check if there are any outliers
    selected_columns = [column for column in df.columns if column.startswith("is_outlier")]

    # Adding all the outlier columns into a new colum "total_outliers", to see the total number of outliers
    df = df.withColumn('total_outliers',sum(df[column] for column in selected_columns))

    # Dropping the extra columns created above, just to create nice dataframe., without extra columns
    df = df.drop(*[column for column in df.columns if column.startswith("is_outlier")])

    return df


from translate import Translator
from pprint import pprint
from more_itertools import locate
import translators as ts
import jieba
import jieba.posseg as pseg
from pyspark.sql.window import Window

def translate(text):
    from translate import Translator
    translator= Translator(to_lang="Chinese")
    return translator.translate(text)

def find_indices(l, item_to_find):
    return locate(l, lambda x: x == item_to_find)

def get_token_tag(text):
    words = pseg.cut(text, use_paddle=True) #paddle模式
    w =[]
    t =[]
    for a, b in words:
        if a != " ":
            w.append(a)
            t.append(b)
    return w,t

def tag_chinese(text):
    w,t = get_token_tag(text)
    wcpy = w.copy()
    
    if "eng" in t:
        idx = find_indices(t, "eng")
        for i in idx:
            try:
                wcpy[i] = translate(" ".join(tokenize_chinese(wcpy[i])))
            except:
                return w,t
            
        nw = "".join(wcpy)
        w2,t2 =  get_token_tag(nw)
        
        if len(w2) == len(w):
            return w, t2
        else:
            return w,t
    else:
        return w,t


udfTagChinese = udf(lambda x : tag_chinese(x), ArrayType(ArrayType(StringType())))


def assign_id_column(df):
    df_mono = df.withColumn("monotonically_increasing_id", monotonically_increasing_id())
    w=Window.orderBy("monotonically_increasing_id")
    return df_mono.withColumn("id",dense_rank().over(w)).drop("monotonically_increasing_id")

