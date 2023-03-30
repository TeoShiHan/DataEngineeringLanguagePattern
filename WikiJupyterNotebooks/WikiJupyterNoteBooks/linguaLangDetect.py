from pyspark.sql.functions import col, udf
from lingua import Language, LanguageDetectorBuilder
languages = [Language.ENGLISH, Language.CHINESE, Language.MALAY]


nonDistDetector = LanguageDetectorBuilder.from_languages(*languages).build()
withDistDetector = LanguageDetectorBuilder.from_languages(*languages).with_minimum_relative_distance(0.25).build()


def lingua_detect_dist_25(text):
    global detector

    try:
        if (detector == None):
            detector = LanguageDetectorBuilder.from_languages(*languages).with_minimum_relative_distance(0.25).build()
        return str(detector.detect_language_of(text))
    
    except:
        return "error"

def lingua_detect(text):
    global detector
    
    try:
        if (detector == None):
            detector = LanguageDetectorBuilder.from_languages(*languages).build()
        return str(detector.detect_language_of(text))
    except:
        return "error"    

detectWithDistUDF = udf(lambda z: lingua_detect_dist_25(z))
detectUDF = udf(lambda z: lingua_detect(z))