import malaya
from pyspark.sql.functions import col, udf, size, split
from pyspark.sql.types import StringType, ArrayType

model = None
def posTagBMBI(text):
    global model
    if model == None:
        model = malaya.pos.transformer(model = 'albert', quantized=False)
        r = model.predict(text)
        return [x[0] for x in r],[x[1] for x in r]
    else:
        r = model.predict(text)
        return [x[0] for x in r],[x[1] for x in r]

posUDF = udf(lambda x : posTagBMBI(x), ArrayType(ArrayType(StringType())))