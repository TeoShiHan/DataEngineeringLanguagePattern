import findspark
findspark.init("/home/pc/TestJupyter/opt/spark-3.3.0/spark-3.3.0-bin-hadoop3")
import pyspark
import random
import os
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"]="/home/pc/TestJupyter/opt/spark-3.3.0/venv-spark/bin/python39"

import sparknlp
spark = sparknlp.start()

