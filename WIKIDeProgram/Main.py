if __name__ == "__main__":
    ## arguments the arguments need to be change
    pyspark_bin_path = "/home/pc/TestJupyter/opt/spark-3.3.0/spark-3.3.0-bin-hadoop3"
    python_path = "/home/pc/TestJupyter/opt/spark-3.3.0/venv-spark/bin/python39"
    app_name = "social media"
    data_path = "/home/pc/data/parsed_data/2parsed-extract-mswiki.txt"
    hdfs_working_path = "hdfs://10.123.51.78:8020/user/wikipedia/"
    malay_stopword_path = "stopwords-ms.json"
    jieba_replace_csv = "replacement.csv"
    hdfsHost = "hdfs://10.123.51.78"
    hdfsPort = "8020"
    hdfs_working_path_file = "/user/wikipedia/"
    hiveWarehouseDirectory = "hdfs://10.123.51.78:8020/user/hive/warehouse"
    hiveTriftServerAddress = "thrift://g2.bigtop.it:9083"
    want_remove_outliers = False


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
    from Alternator import *
    from TokenLanguageReplaceTool import *
    from pyspark.ml.feature import NGram
    from KeyGram import *
    from MiddleKey import *
    from StopwordsFilteringTool import *
    from SeqID import *
    from PosTagBMBI import *
    import matplotlib.pyplot as plt
    import math


    ################################################################################################################
    ## initialize spark session
    os.environ["HADOOP_USER_NAME"] = "hdfs"
    os.environ["PYSPARK_PYTHON"] = python_path
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-avro_2.12:3.3.0 pyspark-shell'
    spark = SparkSession.builder \
        .appName("Spark NLP")\
        .master("local[*]")\
        .config("spark.driver.memory", "70g")\
        .config("spark.sql.warehouse.dir", hiveWarehouseDirectory) \
        .config("spark.sql.catalogImplementation", "hive")\
        .config("hive.metastore.uris", hiveTriftServerAddress) \
        .getOrCreate()
    ################################################################################################################
    ## convert data into avro format
    bm = spark.read.text(data_path).sample(0.05)
    print(bm.count())
    print("================ CONVERTING DATA TO AVRO FORMAT ================")
    write_avro(bm, f"{hdfs_working_path}oriDataBM")
    ################################################################################################################
    ## preprocess data that will affect sentence tokenization
    df = read_avro(spark, f"{hdfs_working_path}oriDataBM")
    pp = pp(df)

    print("================ SAVE THE PREPROCESSED DATA ================")
    write_avro(pp, f"{hdfs_working_path}ppDataBM")
    remove_from_hdfs(f"{hdfs_working_path}oriDataBM")
    # ################################################################################################################
    # ## doing sentence tokenization on the dataset
    df = read_avro(spark, f"{hdfs_working_path}ppDataBM")
    bmSent = get_eng_senDF(df)
    bmSent= bmSent.withColumn("text", explode(bmSent.columns[0])).select("text")

    print("================ SAVE THE SENTENCE TOKENIZE DATA ================")
    write_avro(bmSent, hdfs_working_path+"sent_tokenize_bm.avro")
    remove_from_hdfs(f"{hdfs_working_path}ppDataBM")
    # ################################################################################################################
    # ## preprocessing II
    df = read_avro(spark, f"{hdfs_working_path}sent_tokenize_bm.avro")
    p2 = preprocessing2(df, "text")

    print("================ SAVE THE CLEANED SENTENCE DATA ================")
    write_avro(p2, hdfs_working_path+"p2_bm")
    remove_from_hdfs(f"{hdfs_working_path}sent_tokenize_bm.avro")
    # ################################################################################################################
    # ## preprocessing III
    df = read_avro(spark, f"{hdfs_working_path}p2_bm")
    sen = get_sentence(df, "text").select("text")
    flat = sen.select(flatten(sen.text).alias('text'))
    a = flat.rdd.map(lambda x : (' '.join(x[0]),1)).toDF(["text", "id"]).drop("id")

    print("================ SAVE THE FINAL PROCESS SENTENCE DATA ================")
    write_avro(a, f"{hdfs_working_path}final_process_bm")
    remove_from_hdfs(f"{hdfs_working_path}p2_bm")
    # # ################################################################################################################
    # # ## Word count statistics
    df = read_avro(spark, f"{hdfs_working_path}final_process_bm")
    malayStopWords = set(json.load(open(malay_stopword_path)))
    stopwordsSet = set(stopwords.words('english') + stopwords.words('chinese') ).union(malayStopWords)
    stopWordsRemoval = StopwordsFilteringTool(stopwordsSet)
    nsbm = stopWordsRemoval.get_no_stop_word_df(df, "text")
    wc = \
        nsbm.rdd.flatMap(lambda x : x.text) \
        .map(lambda x : (x,1)).reduceByKey(lambda V1, V2 : V1+V2) \
        .toDF(["word","word_count"])

    wc\
        .select("word","word_count", F.row_number() \
                .over(Window.partitionBy().orderBy(desc(wc['word_count']))) \
                .alias("id")).show(1000)
    # # ################################################################################################################
    # # ## Prompt user to enter the keywords
    # # ## Select topics
    keys = []
    want_choose = True
    while want_choose:
        key = input("enter keyword : ")
        keys.append(key)
        decision = input("Again? (y / n) :  ")
        if not (decision.upper() == 'Y' or decision.upper() == 'YES'):
            want_choose = False
    print("your selected keys is " + str(keys))
    # # ################################################################################################################
    regex = "|".join(keys)
    filtered = df.filter( df.text.rlike(regex))

    print("================ SAVE THE FILTERED DATA ================")
    write_avro(filtered, f"{hdfs_working_path}filtered_bm_data")
    # # ################################################################################################################
    # # ## create sentence word count statistics
    df = read_avro(spark, f"{hdfs_working_path}filtered_bm_data")
    with_statistics = df.withColumn("token", u.tokenize_chineseDF(df.text))
    with_statistics = with_statistics.withColumn("length", size(with_statistics.token))
    df = with_statistics.select("text", "length")

    print("==========SENTENCE WITH WORD COUNT==========")
    write_avro(df, hdfs_working_path+"bm_sentence_with_statistics")
    remove_from_hdfs(hdfs_working_path+"filtered_bm_data")
    # # ################################################################################################################
    # # ## remove the outliers
    s = read_avro(spark, hdfs_working_path+"bm_sentence_with_statistics")

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
    write_avro(noS, hdfs_working_path+"bm_sentence_without_out")
    remove_from_hdfs(hdfs_working_path+"bm_sentence_with_statistics")
    # ################################################################################################################
    # ## pos tag
    df = read_avro(spark, hdfs_working_path+"bm_sentence_without_out")
    df = assign_id_column(df)
    write_avro(df, hdfs_working_path+"bm_with_id")
    df = read_avro(spark, hdfs_working_path+"bm_with_id")

    numberOfRows = df.count()
    splitInto = math.ceil(df.count()/1000)
    segments = get_segments(df, splitInto)



    # split to smaller file to easier processing
    for i in range(splitInto):
        print("write " + str(i))
        write_avro(segments[i].select("text"), f"{hdfs_working_path}IN/{i}")
        print("LEN = " + str(segments[i].count()))

    print(f"the file is splitted into {splitInto} files to ease the prcessing")


    for i in range(splitInto):
        df = read_avro(spark, f"{hdfs_working_path}IN/{i}")
        df.show(5, False)
        tag = df.withColumn("tag", posUDF(df.text))
        write_avro(tag, f"{hdfs_working_path}OUT/{i}")
        df.unpersist()
        print("done execution !")    

        remove_from_hdfs(f"{hdfs_working_path}IN")

    ## merge the file to one and remove the out directory
    alt = Alternator()


    for i in range(splitInto):
        if i == 0:
            merge = read_avro(spark, f"{hdfs_working_path}OUT/{i}")
            write_avro(merge, f"{hdfs_working_path}Merge/{alt.num}")
        else:
            merge = read_avro(spark, f"{hdfs_working_path}Merge/{alt.num}")
            df = read_avro(spark, f"{hdfs_working_path}OUT/{i}")
            union = unionAll(merge, df)
            alt.alternate()
            write_avro(union, f"{hdfs_working_path}Merge/{alt.num}")
            remove_from_hdfs(f"{hdfs_working_path}Merge/{alt.get_alternate()}")

    remove_from_hdfs(f"{hdfs_working_path}OUT")
    df = read_avro(spark, f"{hdfs_working_path}Merge/{alt.num}")
    df.show(10, False)
    # ################################################################################################################
    # Standardize the format
    df = df.rdd.map(lambda x : (x.text, x.tag[0], x.tag[1])).\
            toDF(["original","token","tag"])
    write_avro(df, f"{hdfs_working_path}BM_tokenAndTag")
    # ################################################################################################################
    ## create n-gram
    a = read_avro(spark, hdfs_working_path+"BM_tokenAndTag")
    a = a.repartition(60)
    a= a.withColumn("sentence_id", F.monotonically_increasing_id())
    a = a.rdd.map(lambda x : (x.sentence_id," ".join(x.token), x.token, x.tag)).toDF(["sentence_id", "original","token","tag"])
    write_avro(a, hdfs_working_path+"bm_sentence_with_tagging_format")
    remove_from_hdfs(hdfs_working_path+"BM_tokenAndTag")
    # ################################################################################################################
    a = read_avro(spark, hdfs_working_path+"bm_sentence_with_tagging_format")
    # ################################################################################################################
    # ## bigram
    tokenNgram =NGram(n=2, inputCol="token", outputCol="token_gram")
    tagNgram =NGram(n=2, inputCol="tag", outputCol="tag_gram")
    n2 = tokenNgram.transform(a)
    n2 = tagNgram.transform(n2)
    t = n2.withColumn("tmp", arrays_zip("token_gram", "tag_gram"))\
    .select("sentence_id", "original", "tag","token", F.posexplode(col("tmp")))\
    .select("sentence_id", "original", "tag", "token", "pos", col("col.token_gram"), col("col.tag_gram"))

    print("==========CREATED BIGRAM==========")
    write_avro(t, hdfs_working_path+"BM_N2_GRAM")
    # ################################################################################################################
    # ## trigram
    tokenNgram =NGram(n=3, inputCol="token", outputCol="token_gram")
    tagNgram =NGram(n=3, inputCol="tag", outputCol="tag_gram")

    n2 = tokenNgram.transform(a)
    n2 = tagNgram.transform(n2)

    t = n2.withColumn("tmp", arrays_zip("token_gram", "tag_gram"))\
    .select("sentence_id", "original", "tag","token", F.posexplode(col("tmp")))\
    .select("sentence_id", "original", "tag", "token", "pos", col("col.token_gram"), col("col.tag_gram"))


    print("==========CREATED TRIGRAM==========")
    write_avro(t, hdfs_working_path+"BM_N3_GRAM")
    # ################################################################################################################
    ## n4 gram
    tokenNgram =NGram(n=4, inputCol="token", outputCol="token_gram")
    tagNgram =NGram(n=4, inputCol="tag", outputCol="tag_gram")

    n2 = tokenNgram.transform(a)
    n2 = tagNgram.transform(n2)

    t = n2.withColumn("tmp", arrays_zip("token_gram", "tag_gram"))\
    .select("sentence_id", "original", "tag","token", F.posexplode(col("tmp")))\
    .select("sentence_id", "original", "tag", "token", "pos", col("col.token_gram"), col("col.tag_gram"))


    print("==========CREATED N4 GRAM==========")
    write_avro(t, hdfs_working_path+"BM_N4_GRAM")
    # ################################################################################################################
    ## n5 gram
    tokenNgram =NGram(n=5, inputCol="token", outputCol="token_gram")
    tagNgram =NGram(n=5, inputCol="tag", outputCol="tag_gram")

    n2 = tokenNgram.transform(a)
    n2 = tagNgram.transform(n2)

    t = n2.withColumn("tmp", arrays_zip("token_gram", "tag_gram"))\
    .select("sentence_id", "original", "tag","token", F.posexplode(col("tmp")))\
    .select("sentence_id", "original", "tag", "token", "pos", col("col.token_gram"), col("col.tag_gram"))
    
    print("==========CREATED N5 GRAM==========")
    write_avro(t, hdfs_working_path+"BM_N5_GRAM")
    # ################################################################################################################
    # ## merge ngram
    n2 = read_avro(spark, hdfs_working_path+"BM_N2_GRAM")
    n3 = read_avro(spark, hdfs_working_path+"BM_N3_GRAM")
    n4 = read_avro(spark, hdfs_working_path+"BM_N4_GRAM")
    n5 = read_avro(spark, hdfs_working_path+"BM_N5_GRAM")
    
    n2 = n2.withColumn("gram_type", lit(2))
    n3 = n3.withColumn("gram_type", lit(3))
    n4 = n4.withColumn("gram_type", lit(4))
    n5 = n5.withColumn("gram_type", lit(5))
    merge = unionAll(n2, n3, n4, n5)
    
    print("==========MERGED THE NGRAMS==========")
    write_avro(merge, hdfs_working_path+"BM_MERGE_GRAM")
    # ################################################################################################################
    # ## gram frequency statistics
    # ## Tag frequency
    # ################################################################################################################
    # ## filter n gram with key words
    df = read_avro(spark, hdfs_working_path+"BM_MERGE_GRAM")
    keys = set(keys)
    keyGramFilter = KeyGram(keys)
    withKeyOnly = keyGramFilter.getDFwithKeywordsOnly(df)

    print("==========GET NGRAM WITH KEYWORDS ONLY==========")
    write_avro(withKeyOnly, f"{hdfs_working_path}BM_KeyOnlyData")
    # ################################################################################################################
    # create the column to identify whether keyword at middle
    df = read_avro(spark,  f"{hdfs_working_path}BM_KeyOnlyData")
    mid = MiddleKey(keys)
    withMiddleFlag = df.withColumn("middle_key", mid.middleUDF(df.gram_type, df.token_gram))

    print("==========MARK THE NGRAM WITH KEYWORDS MIDDLE ON IT==========")
    write_avro(withMiddleFlag, f"{hdfs_working_path}BM_MiddleFlagData")
    remove_from_hdfs(f"{hdfs_working_path}BM_KeyOnlyData")
    # ################################################################################################################
    # ## Frequency statistics
    # ## tag frequency
    df = read_avro(spark,  f"{hdfs_working_path}BM_MiddleFlagData")
    tagFreq = df.groupBy("tag_gram").count().withColumnRenamed("count", "tag_gram_f")
    write_avro(tagFreq, hdfs_working_path+"BM_TAG_FREQ")
    tagFreq = read_avro(spark, hdfs_working_path+"BM_TAG_FREQ")
    withtagFBI = df.join(tagFreq, df.tag_gram == tagFreq.tag_gram).drop(tagFreq.tag_gram)

    print("==========COMPUTED THE TAG FREQUENCY==========")
    write_avro(withtagFBI, hdfs_working_path+"BM_TAG_F")
    remove_from_hdfs(hdfs_working_path+"BM_TAG_FREQ")
    # ################################################################################################################
    # ## Token frequency
    df = read_avro(spark, hdfs_working_path+"BM_TAG_F")
    bitokenFreq = df.groupBy("token_gram").count().withColumnRenamed("count", "token_gram_f")
    write_avro(bitokenFreq, hdfs_working_path+"BM_TOKEN_FREQ")
    tokenFreq = read_avro(spark, hdfs_working_path+"BM_TOKEN_FREQ")
    biwithTokF = df.join(tokenFreq, df.token_gram == tokenFreq.token_gram).drop(tokenFreq.token_gram)

    print("==========COMPUTED THE TOKEN FREQUENCY==========")
    write_avro(biwithTokF, hdfs_working_path+"BM_TOKEN_F")
    remove_from_hdfs(hdfs_working_path+"BM_TOKEN_FREQ")
    remove_from_hdfs(hdfs_working_path+"BM_TAG_F")
    # ################################################################################################################
    # ## Table Normalization
    # ## Gram table
    df = read_avro(spark, hdfs_working_path+"BM_TOKEN_F")
    df = df.repartition(800)
    gram_table = df.select("sentence_id", "pos", "token_gram", "token_gram_f", "tag_gram", "tag_gram_f", "gram_type", "containsKey", "middle_key")
    gram_table = gram_table.withColumn("uniqueID", monotonically_increasing_id())
    write_avro(gram_table, hdfs_working_path+"BM_GRAM_TABLE")
    # ################################################################################################################
    # ## Source Table
    originalSentence  = df.dropDuplicates((['sentence_id'])).select("sentence_id", "original", "tag", "token")
    originalSentence = originalSentence.repartition(800)
    print("if disk block error happen, try increase the value of repartition")
    originalSentence = originalSentence.withColumn("uniqueID", monotonically_increasing_id())
    write_avro(originalSentence, hdfs_working_path+"BM_SOURCE_TABLE")
    # ################################################################################################################
    # ## Into Hive
    gram_table = read_avro(spark, hdfs_working_path+"BM_GRAM_TABLE")
    original_table = read_avro(spark, hdfs_working_path+"BM_SOURCE_TABLE")

    spark.sql("create database if not exists wikipedia_db;")
    spark.sql("use wikipedia_db;")
    spark.sql("drop table if exists gram_table_ms;")
    spark.sql("drop table if exists source_data_ms;")


    gram_table.write\
    .format("orc") \
    .mode("overwrite") \
    .saveAsTable("wikipedia_db.gram_table_ms")


    original_table.write\
    .format("orc") \
    .mode("overwrite") \
    .saveAsTable("wikipedia_db.source_data_ms")

    # ################################################################################################################
    # ## Analytics
    # # analytics : Top 10 Common Pos Pattern
    Q="""
    select
        distinct tag_gram,
        tag_gram_f
    from
        gram_table_ms
    order by
        tag_gram_f DESC limit 10;
    """

    SMPos = spark.sql(Q).toPandas()

    fig, ax = plt.subplots(1,1,figsize=(15,5))

    x = SMPos['tag_gram'][:10].tolist()
    y = SMPos['tag_gram_f'][:10].tolist()
    ax.barh(x,y)
    ax.set_title('Top 10 Most Common POS Patterns in Wikipedia Malay')
    for index, value in enumerate(y):
        ax.text(value, index,
                 str(value))
    ax.invert_yaxis()

    try:
        shutil.rmtree("diagram")
    except:
        pass
    
    plt.savefig('top10 common pos pattern.png', dpi=300, bbox_inches='tight')
    # ################################################################################################################
    # ## Analytics
    # # analytics : Top 10 Common Token Pattern
    Q="""
    select
        distinct token_gram,
        token_gram_f
    from
        gram_table_ms
    order by
        token_gram_f DESC limit 10;
    """

    SMtok = spark.sql(Q).toPandas()

    fig, ax = plt.subplots(1,1,figsize=(15,5))

    x = SMtok['token_gram'][:10].tolist()
    y = SMtok['token_gram_f'][:10].tolist()
    ax.barh(x,y)
    ax.set_title('Top 10 Most Common Token Patterns in Wikipedia Malay')
    for index, value in enumerate(y):
        ax.text(value, index,
                 str(value))
    ax.invert_yaxis()

    plt.show()