from pyspark.sql.functions import col, udf
from lingua import Language, LanguageDetectorBuilder
languages =[Language.CHINESE, Language.MALAY, Language.ENGLISH]

def lingua_detect(text):
    try:
        detector = LanguageDetectorBuilder.from_languages(*languages).build()
        return str(detector.detect_language_of(text))
    except:
        return "error"

detectUDF = udf(lambda z: lingua_detect(z))