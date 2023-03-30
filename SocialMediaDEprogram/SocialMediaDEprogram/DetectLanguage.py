from pyspark.sql.functions import col, udf
from lingua import Language, LanguageDetectorBuilder
languages = [Language.ENGLISH, Language.CHINESE, Language.MALAY]


nonDistDetector = LanguageDetectorBuilder.from_languages(*languages).build()
withDistDetector = LanguageDetectorBuilder.from_languages(*languages).with_minimum_relative_distance(0.05).build()


def lingua_detect_dist(text):
    global withDistDetector

    try:
        if (withDistDetector == None):
            withDistDetector = LanguageDetectorBuilder.from_languages(*languages).with_minimum_relative_distance(0.05).build()
        return str(withDistDetector.detect_language_of(text))
    
    except:
        return "error"

def lingua_detect(text):
    global nonDistDetector
    
    try:
        if (nonDistDetector == None):
            nonDistDetector = LanguageDetectorBuilder.from_languages(*languages).build()
        return str(nonDistDetector.detect_language_of(text))
    except:
        return "error"    

detectWithDistUDF = udf(lambda z: lingua_detect_dist(z))
detectUDF = udf(lambda z: lingua_detect(z))