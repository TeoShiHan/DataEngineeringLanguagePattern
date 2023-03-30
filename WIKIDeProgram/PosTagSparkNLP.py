import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf, col
import os

document_assembler = DocumentAssembler() \
.setInputCol("sentence") \
.setOutputCol("document")

sentence_detector = SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")

posTagger = PerceptronModel.pretrained("pos_ud_ewt", "en") \
.setInputCols(["document", "token"]) \
.setOutputCol("pos")

pipeline = Pipeline(stages=[
document_assembler,
sentence_detector,
tokenizer,
posTagger])