
if __name__ == "__main__":
    ## arguments
    pyspark_bin_path = "/home/pc/TestJupyter/opt/spark-3.3.0/spark-3.3.0-bin-hadoop3"
    python_path = "/home/pc/TestJupyter/opt/spark-3.3.0/venv-spark/bin/python39"
    app_name = "social media"
    data_path = "/home/pc/data/parsed_data/1parsed-extract-enwiki.txt"
    hdfs_working_path = "hdfs://10.123.51.78:8020/user/wikipedia/"
    malay_stopword_path = "stopwords-ms.json"
    jieba_replace_csv = "replacement.csv"
    hdfsHost = "hdfs://10.123.51.78"
    hdfsPort = "8020"
    hdfs_working_path_file = "/user/wikipedia/"
    hiveWarehouseDirectory = "hdfs://10.123.51.78:8020/user/hive/warehouse"
    hiveTriftServerAddress = "thrift://g2.bigtop.it:9083"
    want_remove_outliers= False

    import shutil
    import os
    import json
    import findspark
    findspark.init(pyspark_bin_path)
    from FileIO import *
    import utilities as u
    import pyspark
    from Preprocessing import *
    from pyspark.sql import SparkSession
    from DetectLanguage import *
    from LangAgg import *
    from WordCount import *
    from pyspark.sql.window import Window
    from nltk.corpus import stopwords
    from FilterOutliers import *
    from postagChinese import *
    from JiebaTagToStandardTag import *
    from SegmentDataframe import *
    from PosTagBMBI import *
    from Alternator import *
    from TokenLanguageReplaceTool import *
    from pyspark.ml.feature import NGram
    from KeyGram import *
    from MiddleKey import *
    from StopwordsFilteringTool import *
    import matplotlib.pyplot as plt
    import math


    ################################################################################################################
    ## initialize spark session
    os.environ["HADOOP_USER_NAME"] = "hdfs"
    os.environ["PYSPARK_PYTHON"] = python_path
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-avro_2.12:3.3.0,com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.0 pyspark-shell'


    spark = SparkSession.builder \
        .appName("Spark NLP")\
        .master("local[*]")\
        .config("spark.driver.memory", "70g")\
        .config("spark.sql.warehouse.dir", hiveWarehouseDirectory) \
        .config("spark.sql.catalogImplementation", "hive")\
        .config("hive.metastore.uris", hiveTriftServerAddress) \
        .getOrCreate()

    from PosTagSparkNLP import *


    ################################################################################################################
    ## convert data into avro format
    bi = spark.read.text(data_path)
    print("================ CONVERTING DATA TO AVRO FORMAT ================")
    write_avro(bi, f"{hdfs_working_path}oriDataBI")
    ################################################################################################################
    ## preprocess data that will affect sentence tokenization
    df = read_avro(spark, f"{hdfs_working_path}oriDataBI")
    pp = pp(df)

    print("================ SAVE THE PREPROCESSED DATA ================")
    write_avro(pp, f"{hdfs_working_path}ppDataBI")
    remove_from_hdfs(f"{hdfs_working_path}oriDataBI")
    ################################################################################################################
    ## doing sentence tokenization on the dataset
    df = read_avro(spark, f"{hdfs_working_path}ppDataBI")
    engSent = get_eng_senDF(df)
    engSent= engSent.withColumn("text", explode(engSent.columns[0])).select("text")

    print("================ SAVE THE SENTENCE TOKENIZE DATA ================")
    write_avro(engSent, hdfs_working_path+"sent_tokenize_bi.avro")
    remove_from_hdfs(f"{hdfs_working_path}ppDataBI")
    ################################################################################################################
    ## preprocessing II
    df = read_avro(spark, f"{hdfs_working_path}sent_tokenize_bi.avro")
    p2 = preprocessing2(df, "text")

    print("================ SAVE THE CLEANED SENTENCE DATA ================")
    write_avro(p2, hdfs_working_path+"p2_bi")
    remove_from_hdfs(f"{hdfs_working_path}sent_tokenize_bi.avro")
    ################################################################################################################
    ## preprocessing III
    df = read_avro(spark, f"{hdfs_working_path}p2_bi")
    sen = get_sentence(df, "text").select("text")
    flat = sen.select(flatten(sen.text).alias('text'))
    a = flat.rdd.map(lambda x : (' '.join(x[0]),1)).toDF(["text", "id"]).drop("id")

    print("================ SAVE THE FINAL PROCESS SENTENCE DATA ================")
    write_avro(a, f"{hdfs_working_path}final_process_bi")
    remove_from_hdfs(f"{hdfs_working_path}p2_bi")
    ################################################################################################################
    ## Word count statistics
    malayStopWords = set(json.load(open(malay_stopword_path)))
    stopwordsSet = set(stopwords.words('english') + stopwords.words('chinese') ).union(malayStopWords)

    df = read_avro(spark, f"{hdfs_working_path}final_process_bi")

    stopWordsRemoval = StopwordsFilteringTool(stopwordsSet)

    nsbi = stopWordsRemoval.get_no_stop_word_df(df, "text")

    wc = \
        nsbi.rdd.flatMap(lambda x : x.text) \
        .map(lambda x : (x,1)).reduceByKey(lambda V1, V2 : V1+V2) \
        .toDF(["word","word_count"])

    wc\
        .select("word","word_count", F.row_number() \
                .over(Window.partitionBy().orderBy(desc(wc['word_count']))) \
                .alias("id")).show(1000)
    ################################################################################################################
    ## Prompt user to enter the keywords
    ## Select topics
    keys = []
    want_choose = True
    while want_choose:
        key = input("enter keyword : ")
        keys.append(key)
        decision = input("Again? (y / n) :  ")
        if not (decision.upper() == 'Y' or decision.upper() == 'YES'):
            want_choose = False
    print("your selected keys is " + str(keys))
    ################################################################################################################
    regex = "|".join(keys)
    filtered = df.filter( df.text.rlike(regex))

    print("================ SAVE THE FILTERED DATA ================")
    write_avro(filtered, f"{hdfs_working_path}filtered_bi_data")
    ################################################################################################################
    ## create sentence word count statistics
    df = read_avro(spark, f"{hdfs_working_path}filtered_bi_data")
    with_statistics = df.withColumn("token", u.tokenize_chineseDF(df.text))
    with_statistics = with_statistics.withColumn("length", size(with_statistics.token))
    df = with_statistics.select("text", "length")

    print("==========SENTENCE WITH WORD COUNT==========")
    write_avro(df, hdfs_working_path+"bi_sentence_with_statistics")
    remove_from_hdfs(hdfs_working_path+"filtered_bi_data")
    ################################################################################################################
    ## remove the outliers
    s = read_avro(spark, hdfs_working_path+"bi_sentence_with_statistics")
    if want_remove_outliers:
        s = s.coalesce(74)
        sC = s.select(s.length)
        sC.toPandas().plot.box()
        plt.show()
        s_detect = find_outliers(s)
        noS = s_detect.filter((s_detect.total_outliers < 1)  & (s_detect.length > 3))
        sC = noS.select(noS.length)
        sC.toPandas().plot.box()
        plt.show()
        print("==========SENTENCE WITHOUT OUTLIERS==========")

    else:
        noS = s
        
    write_avro(noS, hdfs_working_path+"sentence_without_out")
    remove_from_hdfs(hdfs_working_path+"bi_sentence_with_statistics")
    ################################################################################################################
    ## pos tag
    df = read_avro(spark, hdfs_working_path+"sentence_without_out")
    df = df.withColumnRenamed("text", "sentence")
    result = pipeline.fit(df).transform(df)
    final = result.selectExpr("token.result as token","pos.result as pos")

    print("==========BI SENTENCES WITH TAGGING==========")
    write_avro(final, hdfs_working_path+"bi_sentence_with_tagging")
    ################################################################################################################
    ## create n-gram
    a = read_avro(spark, hdfs_working_path+"bi_sentence_with_tagging")
    a = a.repartition(60)
    a= a.withColumn("sentence_id", F.monotonically_increasing_id())
    a = a.rdd.map(lambda x : (x.sentence_id," ".join(x.token), x.token, x.pos)).toDF(["sentence_id", "original","token","tag"])
    write_avro(a, hdfs_working_path+"bi_sentence_with_tagging_format")
    remove_from_hdfs(hdfs_working_path+"bi_sentence_with_tagging")
    ################################################################################################################
    a = read_avro(spark, hdfs_working_path+"bi_sentence_with_tagging_format")
    ################################################################################################################
    ## bigram
    tokenNgram =NGram(n=2, inputCol="token", outputCol="token_gram")
    tagNgram =NGram(n=2, inputCol="tag", outputCol="tag_gram")
    n2 = tokenNgram.transform(a)
    n2 = tagNgram.transform(n2)
    t = n2.withColumn("tmp", arrays_zip("token_gram", "tag_gram"))\
    .select("sentence_id", "original", "tag","token", F.posexplode(col("tmp")))\
    .select("sentence_id", "original", "tag", "token", "pos", col("col.token_gram"), col("col.tag_gram"))

    print("==========CREATED BIGRAM==========")
    write_avro(t, hdfs_working_path+"BI_N2_GRAM")
    ################################################################################################################
    ## trigram
    tokenNgram =NGram(n=3, inputCol="token", outputCol="token_gram")
    tagNgram =NGram(n=3, inputCol="tag", outputCol="tag_gram")

    n2 = tokenNgram.transform(a)
    n2 = tagNgram.transform(n2)

    t = n2.withColumn("tmp", arrays_zip("token_gram", "tag_gram"))\
    .select("sentence_id", "original", "tag","token", F.posexplode(col("tmp")))\
    .select("sentence_id", "original", "tag", "token", "pos", col("col.token_gram"), col("col.tag_gram"))


    print("==========CREATED TRIGRAM==========")
    write_avro(t, hdfs_working_path+"BI_N3_GRAM")
    ################################################################################################################
    ## n4 gram
    tokenNgram =NGram(n=4, inputCol="token", outputCol="token_gram")
    tagNgram =NGram(n=4, inputCol="tag", outputCol="tag_gram")

    n2 = tokenNgram.transform(a)
    n2 = tagNgram.transform(n2)

    t = n2.withColumn("tmp", arrays_zip("token_gram", "tag_gram"))\
    .select("sentence_id", "original", "tag","token", F.posexplode(col("tmp")))\
    .select("sentence_id", "original", "tag", "token", "pos", col("col.token_gram"), col("col.tag_gram"))


    print("==========CREATED N4 GRAM==========")
    write_avro(t, hdfs_working_path+"BI_N4_GRAM")
    ################################################################################################################
    ## n5 gram
    tokenNgram =NGram(n=5, inputCol="token", outputCol="token_gram")
    tagNgram =NGram(n=5, inputCol="tag", outputCol="tag_gram")

    n2 = tokenNgram.transform(a)
    n2 = tagNgram.transform(n2)

    t = n2.withColumn("tmp", arrays_zip("token_gram", "tag_gram"))\
    .select("sentence_id", "original", "tag","token", F.posexplode(col("tmp")))\
    .select("sentence_id", "original", "tag", "token", "pos", col("col.token_gram"), col("col.tag_gram"))


    print("==========CREATED N5 GRAM==========")
    write_avro(t, hdfs_working_path+"BI_N5_GRAM")
    ################################################################################################################
    ## merge ngram
    n2 = read_avro(spark, hdfs_working_path+"BI_N2_GRAM")
    n3 = read_avro(spark, hdfs_working_path+"BI_N3_GRAM")
    n4 = read_avro(spark, hdfs_working_path+"BI_N4_GRAM")
    n5 = read_avro(spark, hdfs_working_path+"BI_N5_GRAM")


    n2 = n2.withColumn("gram_type", lit(2))
    n3 = n3.withColumn("gram_type", lit(3))
    n4 = n4.withColumn("gram_type", lit(4))
    n5 = n5.withColumn("gram_type", lit(5))


    merge = unionAll(n2, n3, n4, n5)

    print("==========MERGED THE NGRAMS==========")
    write_avro(merge, hdfs_working_path+"BI_MERGE_GRAM")
    ################################################################################################################
    ## gram frequency statistics
    ## Tag frequency
    ################################################################################################################
    ## filter n gram with key words
    df = read_avro(spark, hdfs_working_path+"BI_MERGE_GRAM")
    keys = set(keys)
    keyGramFilter = KeyGram(keys)
    withKeyOnly = keyGramFilter.getDFwithKeywordsOnly(df)

    print("==========GET NGRAM WITH KEYWORDS ONLY==========")
    write_avro(withKeyOnly, f"{hdfs_working_path}BI_KeyOnlyData")
    ################################################################################################################
    # create the column to identify whether keyword at middle
    df = read_avro(spark,  f"{hdfs_working_path}BI_KeyOnlyData")
    mid = MiddleKey(keys)
    withMiddleFlag = df.withColumn("middle_key", mid.middleUDF(df.gram_type, df.token_gram))

    print("==========MARK THE NGRAM WITH KEYWORDS MIDDLE ON IT==========")
    write_avro(withMiddleFlag, f"{hdfs_working_path}BI_MiddleFlagData")
    remove_from_hdfs(f"{hdfs_working_path}BI_KeyOnlyData")
    ################################################################################################################
    ## Frequency statistics
    ## tag frequency
    df = read_avro(spark,  f"{hdfs_working_path}BI_MiddleFlagData")
    tagFreq = df.groupBy("tag_gram").count().withColumnRenamed("count", "tag_gram_f")
    write_avro(tagFreq, hdfs_working_path+"BI_TAG_FREQ")
    tagFreq = read_avro(spark, hdfs_working_path+"BI_TAG_FREQ")
    withtagFBI = df.join(tagFreq, df.tag_gram == tagFreq.tag_gram).drop(tagFreq.tag_gram)

    print("==========COMPUTED THE TAG FREQUENCY==========")
    write_avro(withtagFBI, hdfs_working_path+"BI_TAG_F")
    remove_from_hdfs(hdfs_working_path+"BI_TAG_FREQ")
    ################################################################################################################
    ## Token frequency
    df = read_avro(spark, hdfs_working_path+"BI_TAG_F")
    bitokenFreq = df.groupBy("token_gram").count().withColumnRenamed("count", "token_gram_f")
    write_avro(bitokenFreq, hdfs_working_path+"BI_TOKEN_FREQ")
    tokenFreq = read_avro(spark, hdfs_working_path+"BI_TOKEN_FREQ")
    biwithTokF = df.join(tokenFreq, df.token_gram == tokenFreq.token_gram).drop(tokenFreq.token_gram)

    print("==========COMPUTED THE TOKEN FREQUENCY==========")
    write_avro(biwithTokF, hdfs_working_path+"BI_TOKEN_F")
    remove_from_hdfs(hdfs_working_path+"BI_TOKEN_FREQ")
    remove_from_hdfs(hdfs_working_path+"BI_TAG_F")
    ################################################################################################################
    ## Table Normalization
    ## Gram table
    df = read_avro(spark, hdfs_working_path+"BI_TOKEN_F")
    gram_table = df.select("sentence_id", "pos", "token_gram", "token_gram_f", "tag_gram", "tag_gram_f", "gram_type", "containsKey", "middle_key")
    gram_table = gram_table.withColumn("uniqueID", monotonically_increasing_id())
    write_avro(gram_table, hdfs_working_path+"BI_GRAM_TABLE")
    ################################################################################################################
    ## Source Table
    originalSentence  = df.dropDuplicates((['sentence_id'])).select("sentence_id", "original", "tag", "token")
    originalSentence = originalSentence.repartition(800)
    print("if disk block error happen, try increase the value of repartition")
    originalSentence = originalSentence.withColumn("uniqueID", monotonically_increasing_id())
    write_avro(originalSentence, hdfs_working_path+"BI_SOURCE_TABLE")
    ################################################################################################################
    ## Into Hive
    gram_table = read_avro(spark, hdfs_working_path+"BI_GRAM_TABLE")
    original_table = read_avro(spark, hdfs_working_path+"BI_SOURCE_TABLE")

    spark.sql("create database if not exists wikipedia_db;")
    spark.sql("use wikipedia_db;")
    spark.sql("drop table if exists gram_table_en;")
    spark.sql("drop table if exists source_data_en;")


    gram_table.write\
    .format("orc") \
    .mode("overwrite") \
    .saveAsTable("wikipedia_db.gram_table_en")


    original_table.write\
    .format("orc") \
    .mode("overwrite") \
    .saveAsTable("wikipedia_db.source_data_en")

    ################################################################################################################
    ## Analytics
    # analytics : Top 10 Common Pos Pattern
    Q="""
    select
        distinct tag_gram,
        tag_gram_f
    from
        gram_table_en
    order by
        tag_gram_f DESC limit 10;
    """

    SMPos = spark.sql(Q).toPandas()

    fig, ax = plt.subplots(1,1,figsize=(15,5))

    x = SMPos['tag_gram'][:10].tolist()
    y = SMPos['tag_gram_f'][:10].tolist()
    ax.barh(x,y)
    ax.set_title('Top 10 Most Common POS Patterns in Wikipedia English')
    for index, value in enumerate(y):
        ax.text(value, index,
                 str(value))
    ax.invert_yaxis()

    import shutil
    try:
        shutil.rmtree("visualization")
    except:
        pass
    import os
    os.mkdir("visualization")
    
    plt.savefig('visualization/top10enWIKIPOSPattern.png', dpi=300, bbox_inches='tight')
    ################################################################################################################
    ## Analytics
    # analytics : Top 10 Common Token Pattern
    Q="""
    select
        distinct token_gram,
        token_gram_f
    from
        gram_table_en
    order by
        token_gram_f DESC limit 10;
    """

    SMtok = spark.sql(Q).toPandas()

    fig, ax = plt.subplots(1,1,figsize=(15,5))

    x = SMtok['token_gram'][:10].tolist()
    y = SMtok['token_gram_f'][:10].tolist()
    ax.barh(x,y)
    ax.set_title('Top 10 Most Common Token Patterns in Wikipedia English')
    for index, value in enumerate(y):
        ax.text(value, index,
                 str(value))
    ax.invert_yaxis()

    plt.savefig('visualization/top10enWIKITokenPattern.png', dpi=300, bbox_inches='tight')
