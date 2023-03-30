import contractions
from pyspark.sql.types import StringType, ArrayType
import chinese_converter
from cleantext import clean
from pyspark.sql.functions import *
from nltk.tokenize import sent_tokenize
from harvesttext import HarvestText
import chinese_converter
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql.functions import col, udf, size, split


noContr = udf(lambda x : remove_contractions(x), StringType())
linebreak_pattern = r"((\r\n\t)|[\n\v\t])+"
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
phone_pattern = r"((?:^|(?<=[^\w)]))(((\+?[01])|(\+\d{2}))[ .-]?)?(\(?\d{3,4}\)?/?[ .-]?)?(\d{3}[ .-]?\d{4})(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?:$|(?=\W)))|\+?\d{4,5}[ .-/]\d{6,9}"
number_pattern = r"(?:^|(?<=[^\w,.]))[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)(?:$|(?=\b))"
key_pattern = r"(\w+)(?=:):"
num = "[0-9]"
symbols = """☺|☻|\||\/|♥|\[|♦|♣|\.|♠|•|◘|○|◙|♂|♀|♪|♫|☼|►|◄|↕|‼|¶|\$|§|▬|↨|↑|↓|→|←|∟|↔|▲|▼|#|%|&|,|-|\'|:|;|<|=|>|@|]|_|`|¢|£|¥|₧|ƒ|ª|º|¿|⌐|¬|½|¼|¡|«|»|░|▒|▓|│|┤|╡|╢|╖|╕|╣|║|╗|╝|╜|╛|┐|└|┴|┬|├|─|┼|╞|╟|╚|╔|╩|╦|╠|═|╬|╧|╨|╤|╥|╙|╘|╒|╓|╫|╪|┘|┌|█|▄|▌|▐|▀|α|ß|Γ|π|Σ|σ|µ|τ|Φ|Ω|δ|∞|φ|ε|∩|≡|±|≥|≤|⌠|⌡|÷|≈|°|\+|∙|·|√|ⁿ|\)|\(|²|■|\~|\.|\?|\*|\!|\✓|\^|？|｡|\。|\|:|;|＂"""


def preprocess_for_token_with_dot_delimeter(df):
    return df.withColumn("text", regexp_replace(df.columns[0], email_pattern, 'EMAIL')) \
             .withColumn("text", regexp_replace(df.columns[0], r'http\S+', ''))         \
             .withColumn("text", regexp_replace(df.columns[0], r'<[^>]+>', ''))  \
             .withColumn("text", regexp_replace(df.columns[0], r'@[\w]+', '')).select("text")


def get_sentenct_token(t):
    return sent_tokenize(t, "english")


engSentence = udf(lambda t:get_sentenct_token(t),ArrayType(StringType()))


def get_eng_senDF(df):
    return df.withColumn("text", engSentence(df.columns[0]))


ht = HarvestText()
def chinese_para_to_sentence(text):
    return ht.cut_sentences(text)

getChiS = udf(lambda x : chinese_para_to_sentence(x), ArrayType(StringType()))


def get_df_chiSen(df):
    return df.withColumn("text", getChiS(df.columns[0]))


def remove_contractions(text):
    return contractions.fix(text)


def clean_punctuations(text):
    return clean(fix_unicode=True, no_punct=True)

def simplifiy_chinese(text):
    return chinese_converter.to_simplified(text)

def preprocessing(df, col_name):
    return  \
    df.withColumn(col_name, lower(col(col_name)))\
    .withColumn(col_name, regexp_replace(col_name, linebreak_pattern, '')) \
    .withColumn(col_name, regexp_replace(col_name, r'@[\w]+', 'ALIAS'))\
    .withColumn(col_name, regexp_replace(col_name, phone_pattern, 'PHONE'))\
    .withColumn(col_name, regexp_replace(col_name, key_pattern, '')) \
    .withColumn(col_name, regexp_replace(col_name, "#([a-zA-Z0-9_]{1,50})",''))\
    .withColumn(col_name, regexp_replace(col_name, r'\s+', ' ')) \
    .withColumn(col_name, regexp_replace(col_name, num, '')) \
    .withColumn(col_name, noContr(col(col_name))) \
    .filter( (df[col_name] != "") | (df[col_name] != None)) \
    .withColumn(col_name, sentences(col(col_name))) \
    .select(flatten(col(col_name)).alias('text')) \
    .rdd.map(lambda x : (" ".join(x.text), 1)).toDF(["text","id"]).drop("id") \
    .withColumn(col_name, regexp_replace(col_name, symbols, '')).filter(col(col_name) != "")


def simplifiy_chinese(text):
    return chinese_converter.to_simplified(text)

udfSimpChinese = udf(lambda x : simplifiy_chinese(x), StringType())