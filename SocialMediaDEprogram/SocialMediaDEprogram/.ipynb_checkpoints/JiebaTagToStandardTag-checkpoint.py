import csv
import re
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql.functions import col, udf, size, split, lit


dict2  = {}
def init_replacement_dict(csv_path):
    global dict2

    with open(csv_path, 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:

        value =row[1].replace("POS-","").upper()
        value =  re.sub("^N$", "NOUN", value)
        value =  re.sub("^V$", "VERB", value)
        value =  re.sub("^UNKNOWN$", "X", value)

        dict2[row[0]] =value


dictionary = {
"a": "ADJ",
"p": "ADP",
"d": "ADV",
"u": "ADX",
"c": "CONJ",
"xc": "DET",
"n": "NOUN",
"r": "PRON",
"nr": "PROPN",
"ns": "PROPN",
"nt": "PROPN",
"nw": "PROPN", 
"nz": "PROPN", 
"w": "SYM",
"v": "VERB",
"w":"PUNCT",
"TIME":"TIME",
"ORG":"ORG",
"PER":"PER",
"LOC":"LOC",
"eng":"X",
"x":"X"
}


def standardize_token(text):
    if text in dictionary:
        return dictionary[text]
    elif text in dict2:
        return dict2[text]
    else:
        return "X"

    
    
def getStdArray(arr):
    return [standardize_token(x) for x in arr]

stdTokenUDF = udf(lambda x : getStdArray(x), ArrayType(StringType()))