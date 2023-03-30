from pyspark.sql.functions import *
from pyspark.sql.types import StringType, ArrayType
from translate import Translator
from pprint import pprint
from more_itertools import locate
import translators as ts
import jieba
import jieba.posseg as pseg
from translate import Translator

# have limit need api
def translate(text):
    translator = Translator(from_lang= "English", to_lang="Chinese")
    Translation = translator.translate(text)

def find_indices(l, item_to_find):
    return locate(l, lambda x: x == item_to_find)

def get_token_tag(text):
    words = pseg.cut(text, use_paddle=True) #paddle模式
    w =[]
    t =[]
    for a, b in words:
        if a != " ":
            w.append(a)
            t.append(b)
    return w,t

# have limitation with translation api
def tag_chinese_with_translate_capability(text):
    r = get_token_tag(text)    
    iw = r[0]
    it = r[1]
    initialLength = len(iw)
    if "eng" in it:
        engIndex = find_indices(it, "eng")
        engWords = []
        
        for i in engIndex:
            engWords.append(iw[i])
        
        try:
            for i in engWords:
                translated = translate(i)
                print("re")
                print(translated)
                text = text.replace(i, translated)
        except:
            return iw,it
        
        
        nr = get_token_tag(text)
        
        lengthAfterTranslate = len(nr[0])
        
        if(lengthAfterTranslate == initialLength):
            return iw, nr[1]
        else:
            return iw,it
    else:   
        return iw,it

udfTranslateTagChinese = udf(lambda x : tag_chinese_with_translate_capability(x), ArrayType(ArrayType(StringType())))
udfTagChinese = udf(lambda x :  get_token_tag(x), ArrayType(ArrayType(StringType())))