from pyspark.sql.functions import *
from pyspark.sql.types import *
import jieba

def rem_stop(text, stopwordsSet):
    token = jieba.cut(text, cut_all=True)
    l = []
    for x in token:
        if x in stopwordsSet or x == "" or x == " ":
            continue
        else:
            l.append(x)
    return l


class StopwordsFilteringTool:
    def __init__(self, stopwordset):
        self.stopwordSet = stopwordset
        self.R_sw = udf(lambda x : self.rem_stop(x), ArrayType(StringType()))
        
    def rem_stop(self, text):
        token = jieba.cut(text, cut_all=True)
        l = []
        for x in token:
            if x in self.stopwordSet or x == "" or x == " ":
                continue
            else:
                l.append(x)
        return l
    
        
    def get_no_stop_word_df(self, df, colname):
        return df.withColumn(colname, self.R_sw(col(colname)))