from pyspark.sql.functions import *
from pyspark.sql.types import *


class MiddleKey:
    def __init__(self, keyset):
        self.keyset = keyset
        self.middleUDF = udf(lambda x,y : self.is_middle(x,y), BooleanType() )
        
    def is_middle(self, gramType, token_gram):
        token_gram = token_gram.split(" ")
        if gramType == "2" or gramType == "4":
            return False
        elif gramType == "3":
            if token_gram[1] in self.keyset:
                return True
            else:
                return False
        elif gramType == "5":
            if token_gram[2] in self.keyset:
                return True
            else:
                return False
        else:
            return False
    
