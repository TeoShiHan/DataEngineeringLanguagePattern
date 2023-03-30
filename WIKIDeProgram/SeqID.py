from pyspark.sql.functions import *
from pyspark.sql.window import Window


def assign_id_column(df):
    df_mono = df.withColumn("monotonically_increasing_id", monotonically_increasing_id())
    w=Window.orderBy("monotonically_increasing_id")
    return df_mono.withColumn("id",dense_rank().over(w)).drop("monotonically_increasing_id")