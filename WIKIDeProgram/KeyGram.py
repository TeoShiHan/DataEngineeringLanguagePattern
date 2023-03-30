from pyspark.sql.functions import *
from pyspark.sql.types import *



class KeyGram:
    def __init__(self, keyset):
        self.keyset = keyset
        self.keyDeter = udf(lambda x : self.contains_keywords(x), BooleanType())

    
    def getDFwithKeywordsOnly(self, df):
        withFlag = df.withColumn("containsKey", self.keyDeter(col("token_gram")))
        withKey = withFlag.filter(col("containsKey") == True)
        return withKey
    
    def contains_keywords(self, text):
        for x in text.split(" "):
            if x in self.keyset:
                return True
            else:
                return False