import jieba

def tokenization(text):
    return [x for x in jieba.cut(text, cut_all=True) if x != "" and " " not in x]

def get_wc_df(df):
    df = \
    df.rdd.flatMap(lambda x : tokenization(x.text))\
      .map(lambda x : (x,1)) \
      .reduceByKey(lambda V1, V2 : V1+V2) \
      .toDF(["word","word_count"])
    return df