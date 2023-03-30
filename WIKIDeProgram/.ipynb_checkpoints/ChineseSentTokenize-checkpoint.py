from pyspark.sql.functions import *
from pyspark.sql.types import *
from harvesttext import HarvestText



ht = HarvestText()
def chinese_para_to_sentence(text):
    return ht.cut_sentences(text)

getChiS = udf(lambda x : chinese_para_to_sentence(x), ArrayType(StringType()))


def get_df_chiSen(df):
    return df.withColumn("text", getChiS(df.columns[0]))