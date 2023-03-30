from pyspark.sql.functions import *

class TokenLanguageReplaceTool:
    def __init__(self, replacement, languages):
        self.replacement = replacement
        self.languages = languages
        self.getLangUDF = udf(lambda x : self.get_lang_arr(x), ArrayType(StringType()))

    def get_lang(self, text):
        for i in range(4):
            if text in self.replacement[i]:
                return self.languages[i]
        return "oov"

    def get_lang_arr(self, arr):
        return [self.get_lang(x) for x in arr]