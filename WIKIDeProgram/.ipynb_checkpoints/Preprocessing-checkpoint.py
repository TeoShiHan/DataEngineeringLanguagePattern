from pyspark.sql.types import StringType, ArrayType
import chinese_converter
from cleantext import clean
from pyspark.sql.functions import *
from nltk.tokenize import sent_tokenize
from harvesttext import HarvestText
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql.functions import col, udf, size, split
import contractions
from pyspark.sql.functions import sentences, lit, flatten
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql.functions import col, udf, size, split
import pyspark.sql.functions as F
from pyspark.sql.functions import sentences, lit, flatten
import chinese_converter


## no braces
## no == b ==
## no category
## no other source

linebreak_pattern = r"((\r\n\t)|[\n\v\t])+"
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
phone_pattern = r"((?:^|(?<=[^\w)]))(((\+?[01])|(\+\d{2}))[ .-]?)?(\(?\d{3,4}\)?/?[ .-]?)?(\d{3}[ .-]?\d{4})(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?:$|(?=\W)))|\+?\d{4,5}[ .-/]\d{6,9}"
number_pattern = r"(?:^|(?<=[^\w,.]))[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)(?:$|(?=\b))"
key_pattern = r"(\w+)(?=:):"
num = "[0-9]"
symbols = """☺|☻|\||\/|♥|\[|♦|♣|\.|♠|•|◘|○|◙|♂|♀|♪|♫|☼|►|◄|↕|‼|¶|\$|§|▬|↨|↑|↓|→|←|∟|↔|▲|▼|#|%|&|,|-|\'|:|;|<|=|>|@|]|_|`|¢|£|¥|₧|ƒ|ª|º|¿|⌐|¬|½|¼|¡|«|»|░|▒|▓|│|┤|╡|╢|╖|╕|╣|║|╗|╝|╜|╛|┐|└|┴|┬|├|─|┼|╞|╟|╚|╔|╩|╦|╠|═|╬|╧|╨|╤|╥|╙|╘|╒|╓|╫|╪|┘|┌|█|▄|▌|▐|▀|α|ß|Γ|π|Σ|σ|µ|τ|Φ|Ω|δ|∞|φ|ε|∩|≡|±|≥|≤|⌠|⌡|÷|≈|°|\+|∙|·|√|ⁿ|\)|\(|²|■|\~|\.|\?|\*|\!|\✓|\^|？|｡|\。|\|:|;|＂"""



num = "[0-9]"
px = "[0-9]px"
p1 = "\=(.{1,}?)\= "
p2 = "Category:.*|Kategori:.*"
p3 = "\*(.*?)(?=[ *| , | .|*|,|.])"
p4 = "\{([^}]+)\}"
p5 = r"http\S+"
p6 = "thumb\|.+?(?= )"
p7 = "([^ ])+ : ([^ ])+ |([^ ])+:([^ ])+"



email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
def preprocess_for_token_with_dot_delimeter(df, col_name):
    return df.withColumn(col_name, regexp_replace(col_name, email_pattern, 'EMAIL')) \
             .withColumn(col_name, regexp_replace(col_name, r'http\S+', ''))         \
             .withColumn(col_name, regexp_replace(col_name, r'<[^>]+>', ''))

def pp(df):
    df = df \
    .withColumn("text", regexp_replace(df.columns[0], f"{p5}", '')).select("text")\
    .withColumn("text", regexp_replace("text", f"{p4}", '')) \
    .withColumn("text", regexp_replace("text", f"{p1}", '')) \
    .withColumn("text", regexp_replace("text", f"{p2}", '')) \
    .withColumn("text", regexp_replace("text", f"{p3}|{p6}|{p7}|[0-9]", '')) \
    .withColumn("text", regexp_replace("text", f"{num}|{px}|colspan|right thumb", '')) \
    .withColumn("text", lower("text")) 
    return preprocess_for_token_with_dot_delimeter(df, "text")


def get_sentenct_token(t):
    return sent_tokenize(t, "english")


engSentence = udf(lambda t:get_sentenct_token(t),ArrayType(StringType()))


def get_eng_senDF(df):
    return df.withColumn("text", engSentence(df.columns[0]))

def remove_contractions(text):
    return contractions.fix(text)

def clean_punctuations(text):
    return clean(text, fix_unicode=True, no_punct=True, to_ascii=False)

def simplifiy_chinese(text):
    return chinese_converter.to_simplified(text)

def preprocessing2(df, col_name):
    df = \
    df.withColumn(col_name, F.lower(col(col_name)))\
    .withColumn(col_name, F.regexp_replace(col_name, linebreak_pattern, '')) \
    .withColumn(col_name, F.regexp_replace(col_name, r'@[\w]+', 'ALIAS'))\
    .withColumn(col_name, F.regexp_replace(col_name, phone_pattern, 'PHONE'))\
    .withColumn(col_name, F.regexp_replace(col_name, key_pattern, '')) \
    .withColumn(col_name, F.regexp_replace(col_name, "#([a-zA-Z0-9_]{1,50})",''))\
    .withColumn(col_name, F.regexp_replace(col_name, r'\s+', ' ')) \
    .withColumn(col_name, F.regexp_replace(col_name, num, '')) \
    .withColumn(col_name, F.regexp_replace(col_name, "px|",''))\
    .withColumn(col_name, noContr(F.col(col_name))) \
    .filter( (df[col_name] != "") | (df[col_name] != None)) \
    .withColumn(col_name, sentences(F.col(col_name))) \
    .select(flatten(F.col(col_name)).alias('text')) \
    .rdd.map(lambda x : (" ".join(x.text), 1)).toDF(["text","id"]).drop("id") \
    .withColumn(col_name, F.regexp_replace(col_name, symbols, '')) \
    .withColumn(col_name,cleanPunctUDF(F.col(col_name))) \
    .withColumn(col_name, udfSimpChinese(F.col(col_name)))
    return df.filter(F.col(col_name) != "")

def get_sentence(df, colname):
    return df.select(sentences(df[colname]).alias("text"))

cleanPunctUDF = udf(lambda x : clean_punctuations(x), StringType())
noContr = udf(lambda x : remove_contractions(x), StringType())



def simplifiy_chinese(text):
    return chinese_converter.to_simplified(text)

udfSimpChinese = udf(lambda x : simplifiy_chinese(x), StringType())