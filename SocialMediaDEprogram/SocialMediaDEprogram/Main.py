## arguments


if __name__ == "__main__":
    pyspark_bin_path = "/home/pc/TestJupyter/opt/spark-3.3.0/spark-3.3.0-bin-hadoop3"
    python_path = "/home/pc/TestJupyter/opt/spark-3.3.0/venv-spark/bin/python39"
    app_name = "social media"
    data_path = "/home/pc/Assignment/SocialMedia/Main/output/sample.csv"
    hdfs_working_path = "hdfs://10.123.51.78:8020/user/social_media/"
    malay_stopword_path = "/home/pc/Assignment/node_modules/stopwords-ms/stopwords-ms.json"
    jieba_replace_csv = "/home/pc/Assignment/SocialMedia/Main/replacement.csv"
    hdfsHost = "hdfs://10.123.51.78"
    hdfsPort = "8020"
    hdfs_working_path_file = "/user/social_media/"
    hiveWarehouseDirectory = "hdfs://10.123.51.78:8020/user/hive/warehouse"
    hiveTriftServerAddress = "thrift://g2.bigtop.it:9083"
    remove_outliers = False

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
    import matplotlib.pyplot as plt
    import math

    ################################################################################################################
    ## initialize spark session
    os.environ["HADOOP_USER_NAME"] = "hdfs"
    os.environ["PYSPARK_PYTHON"] = python_path
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-avro_2.12:3.3.0  pyspark-shell'


    spark = SparkSession.builder \
        .master("local[*]") \
        .appName(app_name) \
        .config("spark.driver.memory", "70g")\
        .config("spark.sql.warehouse.dir", hiveWarehouseDirectory) \
        .config("spark.sql.catalogImplementation", "hive")\
        .config("hive.metastore.uris", hiveTriftServerAddress)\
        .getOrCreate()

    ###################################################################################################################################################################################
    ## convert original data to avro file
    csv = spark.read.csv(data_path, header = 'true')
    try:
        remove_from_hdfs(hdfs_working_path+"original_data.avro", header = 'true')
    except:
        pass
    csv.write.format("avro").mode("overwrite").save(hdfs_working_path+"original_data.avro", header = 'true')
    ###################################################################################################################################################################################
    ## preprocessing I : remove the token that will disturb the sentence tokenization
    df = read_avro(spark, hdfs_working_path+"original_data.avro")
    p1 = preprocess_for_token_with_dot_delimeter(df)

    print("==========DATA WITH REDUCE DELEMITER INTERFERE==========")
    write_avro(p1, hdfs_working_path+"reduce_delimeter_disturbance.avro")
    remove_from_hdfs(hdfs_working_path+"original_data.avro")
    print(">>> saved reduce delimeter and removed original......")
    ###################################################################################################################################################################################
    ## preprocessing II : sentence tokenization
    df = read_avro(spark, hdfs_working_path+"reduce_delimeter_disturbance.avro")

    engSent = get_eng_senDF(df)
    engSent= engSent.withColumn("text", explode(engSent.columns[0])).select("text")
    chiSent = get_df_chiSen(df)
    chiSent = chiSent.withColumn("text", explode(chiSent.columns[0]))

    print("==========DATA WITH SENTENCE TOKENIZATION==========")
    write_avro(chiSent, hdfs_working_path+"sent_tokenize.avro")
    remove_from_hdfs(hdfs_working_path+"reduce_delimeter_disturbance.avro")
    print(">>> saved sent tokenize and remove reduce tokenize......")
    ###################################################################################################################################################################################
    ## preprocessing III : remove unwanted data, language independent
    df = read_avro(spark, hdfs_working_path+"sent_tokenize.avro")
    p3 = preprocessing(df, "text")

    print("==========DATA WITHOUT UNWANTED DATA==========")
    write_avro(p3, hdfs_working_path+"without_unwanted.avro")
    remove_from_hdfs(hdfs_working_path+"sent_tokenize.avro")
    print(">>> saved without unwanted and remove sent tokenize......")
    ###################################################################################################################################################################################
    ## Classify Language
    df = read_avro(spark, hdfs_working_path+"without_unwanted.avro")
    df = df.repartition(74)
    withLanguage = df.withColumn("lang", detectUDF(col("text"))) ## detectUDF is custom made UDF

    print("==========DATA WITH LANGUAGE==========")
    write_avro(withLanguage, hdfs_working_path+"with_language.avro")
    remove_from_hdfs(hdfs_working_path+"without_unwanted.avro")
    print(">>> saved with language and remove unwanted......")
    ###################################################################################################################################################################################
    ## Convert chinese character to simplified version
    df = read_avro(spark, hdfs_working_path+"with_language.avro")
    simpChi = df.withColumn("text", udfSimpChinese(col("text")))

    print("==========CHINESE SIMPLIFIED DATA==========")
    write_avro(simpChi, hdfs_working_path+"chinese_simplified.avro")
    remove_from_hdfs(hdfs_working_path+"with_language.avro")
    print(">>> saved with simplified chinese and remove with language......")
    ###################################################################################################################################################################################
    ## Classify the dataframe to several languages
    df = read_avro(spark, hdfs_working_path+"chinese_simplified.avro")
    agg = LanguageAggregor(df, "lang")
    print("these dataset have languages : " + str(agg.all_language))
    visualize_languages(agg)
    por = None
    for i in agg.all_language:
        if i != "None":
            if por == None:
                por = agg.get_lang_dataframe(i)
            else:
                por = unionAll(por, agg.get_lang_dataframe(i))

    print("==========CHINESE, ENGLISH & MALAY DATA==========")            
    write_avro(por, hdfs_working_path+"bmbcbiOnly")
    remove_from_hdfs(hdfs_working_path+"chinese_simplified.avro")
    print(">>> saved with bmbibc only and remove chinese_simplified......")
    ###################################################################################################################################################################################
    ## Word Count
    df = read_avro(spark, hdfs_working_path+"bmbcbiOnly")
    wc = get_wc_df(df)
    vocabLang =  wc.withColumn('wordLang', detectUDF(col('word')))
    vocabLang \
        .select("word","word_count", row_number() \
        .over(Window.partitionBy() \
                .orderBy(desc(vocabLang['word_count']))) \
        .alias("id"), "wordLang")

    print("==========WORD COUNT WITH STOPWORDS==========")
    write_avro(vocabLang, hdfs_working_path+"vocabLangFull")
    ###################################################################################################################################################################################
    ## Word Count Without Stopwords
    malayStopWords = set(json.load(open(malay_stopword_path)))
    stopwordsSet = set(stopwords.words('english') + stopwords.words('chinese') ).union(malayStopWords)
    reg =   ["^"+x+"$" for x in stopwordsSet]
    reg = "|".join(reg)
    df = read_avro(spark, hdfs_working_path+"vocabLangFull")
    regex = "|".join(list(stopwordsSet)) 
    noStopwords = df.filter(~df.word.rlike(reg))
    noStopwords = noStopwords.select("word","word_count", row_number().over(Window.partitionBy() \
      .orderBy(desc(df['word_count']))).alias("id"))

    print("==========WORD COUNT WITHOUT STOPWORDS==========")
    write_avro(noStopwords, hdfs_working_path+"no_stopwords")
    read_avro(spark, hdfs_working_path+"no_stopwords").show(1000, False)
    ###################################################################################################################################################################################
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
    ###################################################################################################################################################################################
    df = read_avro(spark, hdfs_working_path+"bmbcbiOnly")
    regex = "|".join(keys)
    withKeys = df.filter( df.text.rlike(regex))

    print("==========FILTER SENTENCE BY KEYS==========")
    write_avro(withKeys, hdfs_working_path+"sentence_with_keys")
    ###################################################################################################################################################################################
    df = read_avro(spark, hdfs_working_path+"sentence_with_keys")
    with_statistics = df.withColumn("token", u.tokenize_chineseDF(df.text))
    with_statistics = with_statistics.withColumn("length", size(with_statistics.token))
    df = with_statistics.select("text", "length", "lang")

    print("==========SENTENCE WITH WORD COUNT==========")
    write_avro(df, hdfs_working_path+"sentence_with_statistics")
    remove_from_hdfs(hdfs_working_path+"sentence_with_keys")
    ###################################################################################################################################################################################
    s = read_avro(spark, hdfs_working_path+"sentence_with_statistics")

    if remove_outliers:
        s = s.coalesce(74)

        bc = s.filter( (s.lang == "Language.CHINESE") & (s.length > 3))
        bm = s.filter( (s.lang == "Language.MALAY") &  (s.length > 3))
        bi = s.filter( (s.lang == "Language.ENGLISH") &  (s.length > 3))

        bc = bc.coalesce(74)
        bm = bm.coalesce(74)
        bi = bi.coalesce(74)

        bcC = bc.select(bc.length)
        bmC = bm.select(bm.length)
        biC = bi.select(bi.length)

        print("MALAY BOX PLOT")
        bmC.toPandas().plot.box()
        plt.show()

        print("ENGLISH BOX PLOT")
        biC.toPandas().plot.box()
        plt.show()

        print("CHINESE BOX PLOT")
        bcC.toPandas().plot.box()
        plt.show()
        ###################################################################################################################################################################################
        nobc = None
        nobm = None
        nobi = None

        if bc.count() != 0 :
            obc = find_outliers(bc)
            nobc = obc.filter((obc.total_outliers < 1) & (obc.length > 3))
            bcC = nobc.select(nobc.length)

        if bi.count() != 0 :
            obi = find_outliers(bi)
            nobi = obi.filter((obi.total_outliers < 1)  & (obi.length > 3))
            biC = nobi.select(nobi.length)

        if bm.count() != 0 :
            obm = find_outliers(bm)
            nobm = obm.filter((obm.total_outliers < 1)  & (obm.length > 3))
            bmC = nobm.select(nobm.length)

        x = [nobc, nobi, nobm]
        noNone = [x for x in x if x != None]


        print("MALAY BOX PLOT REMOVE OUTLIER")
        bmC.toPandas().plot.box()
        plt.show()


        print("ENGLISH BOX PLOT REMOVE OUTLIER")
        biC.toPandas().plot.box()
        plt.show()


        print("CHINESE BOX PLOT REMOVE OUTLIER")
        bcC.toPandas().plot.box()
        plt.show()


        df = unionAll(*noNone).drop("token").drop("length").drop("total_outliers")
        print("==========SENTENCE WITHOUT OUTLIERS==========")
    else:
        df = s

    write_avro(df, hdfs_working_path+"sentence_without_out")
    ###################################################################################################################################################################################
    # Pos tagging
    df = read_avro(spark, hdfs_working_path+"sentence_without_out")
    bc = df.filter(df.lang == "Language.CHINESE")
    # Pos tagging chinese
    if bc.count() > 0:
        chineseTag = bc.withColumn("chinese_tag", udfTagChinese(bc.text))

        chineseTag = chineseTag.rdd.map(lambda x : (x.text, x.chinese_tag[0], x.chinese_tag[1], x.lang)).\
            toDF(["original","token","chinese_tag", "language"])

        init_replacement_dict(jieba_replace_csv)

        withNormalTag = chineseTag.withColumn("tag", stdTokenUDF(col("chinese_tag")))\
            .withColumn("language", lit("chinese"))
    ###################################################################################################################################################################################
    # Pos tagging english and malay
    bmbi = df.filter( (df.lang == "Language.ENGLISH") | (df.lang == "Language.MALAY"))

    if bmbi.count() > 0:
        bmbi = u.assign_id_column(bmbi)

    numberOfRows = bmbi.count()
    splitInto = math.ceil(bmbi.count()/1000)
    segments = get_segments(bmbi, splitInto)

    print(f"the file is splitted into {splitInto} files to ease the prcessing")

    ## split to smaller file to easier processing
    for i in range(splitInto):
        write_avro(segments[i], f"{hdfs_working_path}IN/{i}")

    for i in range(splitInto):
        df = read_avro(spark, f"{hdfs_working_path}IN/{i}")
        tag = df.withColumn("tag", posUDF(df.text))
        write_avro(tag, f"{hdfs_working_path}/OUT/{i}")  

    sc = spark.sparkContext
    os.system(f"hdfs dfs -rm -r {hdfs_working_path_file}IN")

    ## merge the file to one and remove the out directory
    alt = Alternator()

    for i in range(splitInto):
        if i == 0:
            merge = read_avro(spark, f"{hdfs_working_path}/OUT/{i}")
            write_avro(merge, f"{hdfs_working_path}/Merge/{alt.num}")
        else:
            merge = read_avro(spark, f"{hdfs_working_path}/Merge/{alt.num}")
            df = read_avro(spark, f"{hdfs_working_path}/OUT/{i}")
            union = unionAll(merge, df)
            alt.alternate()
            write_avro(union, f"{hdfs_working_path}/Merge/{alt.num}")
            remove_from_hdfs(f"{hdfs_working_path}/Merge/{alt.get_alternate()}")

    os.system(f"hdfs dfs -rm -r {hdfs_working_path_file}OUT")        
    df = read_avro(spark, f"{hdfs_working_path}/Merge/{alt.num}")
    ###################################################################################################################################################################################
    df = df.rdd.map(lambda x : (x.text, x.tag[0], x.tag[1], x.lang)).\
            toDF(["original","token","tag", "language"])

    merged = df.unionByName(withNormalTag, allowMissingColumns=True)
    print("==========FRAME WITH TAG AND TOKEN==========")
    write_avro(merged, f"{hdfs_working_path}tokenAndTag")
    os.system(f"hdfs dfs -rm -r {hdfs_working_path_file}Merge")        
    df = read_avro(spark, f"{hdfs_working_path}tokenAndTag")
    print("removed merge and saved token And Tag")
    ###################################################################################################################################################################################
    ## Preparing token with languages label using joining
    words = \
    df.rdd.flatMap(lambda x : x.token) \
    .map(lambda x : (x,1)) \
    .toDF(["words"]) \
    .drop("_2")

    uniqueWords = words.select('words').distinct()

    wL= read_avro(spark, hdfs_working_path+"vocabLangFull")
    wL = wL.drop("word_count").drop("id")

    withPreDetected = uniqueWords.join(wL, wL.word ==  uniqueWords.words,"left").drop("word")

    ## null set is those without languages
    nullSet = withPreDetected.filter(withPreDetected.wordLang.isNull())
    notNullSet = withPreDetected.filter(~withPreDetected.wordLang.isNull())
    corrected = nullSet.withColumn("wordLang", detectUDF(nullSet.words))


    print("==========TOKENS WITH LANGUAGES ASSOCIATED==========")
    write_avro(unionAll(corrected, notNullSet), hdfs_working_path+"replacement_token_lang")
    ###################################################################################################################################################################################
    tokenLang = read_avro(spark, hdfs_working_path+"replacement_token_lang")
    df = read_avro(spark, f"{hdfs_working_path}tokenAndTag")
    tokenLang = tokenLang.withColumn("wordLang" ,regexp_replace("wordLang","None|error", "oov"))
    correct_dict = tokenLang

    engSet = correct_dict.filter(correct_dict.wordLang.rlike("Language.ENGLISH"))
    bmSet =correct_dict.filter(correct_dict.wordLang.rlike("Language.MALAY"))
    bcSet = correct_dict.filter(correct_dict.wordLang.rlike("Language.CHINESE"))
    oovSet = correct_dict.filter(~correct_dict.wordLang.rlike("Language.MALAY|Language.CHINESE|Language.ENGLISH"))

    oov = set(oovSet.rdd.flatMap(lambda x : x).collect())
    bi = set(engSet.rdd.flatMap(lambda x : x).collect())
    bm = set(bmSet.rdd.flatMap(lambda x : x).collect())
    bc = set(bcSet.rdd.flatMap(lambda x : x).collect())

    replacement = [bi, bm, bc, oov]
    languages = ["en","ms","zh","oov"]

    tokenToLanguage = TokenLanguageReplaceTool(replacement, languages)
    withLang = df.withColumn("language_pattern", tokenToLanguage.getLangUDF(col("token")))
    print("==========DATAFRAME WITH LANG, TOKEN AND TAG==========")
    write_avro(withLang, f"{hdfs_working_path}tokenAndTagAndLang")
    remove_from_hdfs(f"{hdfs_working_path}tokenAndTag")
    print("removed token and tag saved token, tag, and lang")
    ###################################################################################################################################################################################
    ## creating N-gram for token, language and tagging
    a = read_avro(spark, f"{hdfs_working_path}tokenAndTagAndLang")
    a = a.withColumn("sentence_id", monotonically_increasing_id())

    ## bigram
    tokenNgram =NGram(n=2, inputCol="token", outputCol="token_gram")
    tagNgram =NGram(n=2, inputCol="tag", outputCol="tag_gram")
    langNgram =NGram(n=2, inputCol="language_pattern", outputCol="lang_gram")

    n2 = tokenNgram.transform(a)
    n2 = tagNgram.transform(n2)
    n2 = langNgram.transform(n2)

    t = n2.withColumn("tmp", arrays_zip("token_gram", "tag_gram", "lang_gram"))\
    .select("sentence_id", "original", "tag","language", "token", "chinese_tag", \
            "language_pattern","token", posexplode(col("tmp")))\
    .select("sentence_id", "original", "tag","language", "token", "chinese_tag", \
            "language_pattern", "pos",col("col.token_gram"), col("col.tag_gram"),  col("col.lang_gram"))

    print("==========CREATED BIGRAM==========")
    write_avro(t, f"{hdfs_working_path}N2_GRAM")
    print("save bigram")
    ###################################################################################################################################################################################
    ## trigram
    tokenNgram =NGram(n=3, inputCol="token", outputCol="token_gram")
    tagNgram =NGram(n=3, inputCol="tag", outputCol="tag_gram")
    langNgram =NGram(n=3, inputCol="language_pattern", outputCol="lang_gram")

    n = tokenNgram.transform(a)
    n = tagNgram.transform(n)
    n = langNgram.transform(n)

    t = n.withColumn("tmp", arrays_zip("token_gram", "tag_gram", "lang_gram"))\
    .select("sentence_id", "original", "tag","language", "token", "chinese_tag", \
            "language_pattern","token", posexplode(col("tmp")))\
    .select("sentence_id", "original", "tag","language", "token", "chinese_tag", \
            "language_pattern", "pos",col("col.token_gram"), col("col.tag_gram"),  col("col.lang_gram"))

    print("==========CREATED TRIGRAM==========")
    write_avro(t, f"{hdfs_working_path}N3_GRAM")
    print("save trigram")
    ###################################################################################################################################################################################
    ## 4 gram
    tokenNgram =NGram(n=4, inputCol="token", outputCol="token_gram")
    tagNgram =NGram(n=4, inputCol="tag", outputCol="tag_gram")
    langNgram =NGram(n=4, inputCol="language_pattern", outputCol="lang_gram")

    n = tokenNgram.transform(a)
    n = tagNgram.transform(n)
    n = langNgram.transform(n)

    t = n.withColumn("tmp", arrays_zip("token_gram", "tag_gram", "lang_gram"))\
    .select("sentence_id", "original", "tag","language", "token", "chinese_tag",\
            "language_pattern","token", posexplode(col("tmp")))\
    .select("sentence_id", "original", "tag","language", "token", "chinese_tag", \
            "language_pattern", "pos",col("col.token_gram"), col("col.tag_gram"),  col("col.lang_gram"))

    print("==========CREATED 4-GRAM==========")
    write_avro(t, f"{hdfs_working_path}N4_GRAM")
    print("save n4 gram")
    ###################################################################################################################################################################################
    ## 5 gram
    tokenNgram =NGram(n=5, inputCol="token", outputCol="token_gram")
    tagNgram =NGram(n=5, inputCol="tag", outputCol="tag_gram")
    langNgram =NGram(n=5, inputCol="language_pattern", outputCol="lang_gram")

    n = tokenNgram.transform(a)
    n = tagNgram.transform(n)
    n = langNgram.transform(n)

    t = n.withColumn("tmp", arrays_zip("token_gram", "tag_gram", "lang_gram"))\
    .select("sentence_id", "original", "tag","language", "token", "chinese_tag",\
            "language_pattern","token", posexplode(col("tmp")))\
    .select("sentence_id", "original", "tag","language", "token", "chinese_tag",\
            "language_pattern", "pos",col("col.token_gram"), col("col.tag_gram"),  col("col.lang_gram"))

    print("==========CREATED 5-GRAM==========")
    write_avro(t, f"{hdfs_working_path}N5_GRAM")
    print("save n5 gram")
    ###################################################################################################################################################################################
    ## assign gram type and merge
    g2 = read_avro(spark, f"{hdfs_working_path}N2_GRAM")
    g3 = read_avro(spark, f"{hdfs_working_path}N3_GRAM")
    g4 = read_avro(spark, f"{hdfs_working_path}N4_GRAM")
    g5 = read_avro(spark, f"{hdfs_working_path}N5_GRAM")

    g2 = g2.withColumn("gram_type", lit("2"))
    g3 = g3.withColumn("gram_type", lit("3"))
    g4 = g4.withColumn("gram_type", lit("4"))
    g5 = g5.withColumn("gram_type", lit("5"))

    merge = unionAll(g2, g3, g4, g5)

    print("==========ASSEMBLE ALL NGRAM==========")
    write_avro(merge, f"{hdfs_working_path}ALL_GRAM")

    remove_from_hdfs(f"{hdfs_working_path}N2_GRAM")
    remove_from_hdfs(f"{hdfs_working_path}N3_GRAM")
    remove_from_hdfs(f"{hdfs_working_path}N4_GRAM")
    remove_from_hdfs(f"{hdfs_working_path}N5_GRAM")

    print("remove single gram and save merge gram")
    ###################################################################################################################################################################################
    # get the n-gram that contain the keywords only
    df = read_avro(spark, f"{hdfs_working_path}ALL_GRAM")
    keys = set(keys)
    keyGramFilter = KeyGram(keys)
    withKeyOnly = keyGramFilter.getDFwithKeywordsOnly(df)
    print("==========GET NGRAM WITH KEYWORDS ONLY==========")
    write_avro(withKeyOnly, f"{hdfs_working_path}KeyOnlyData")
    ###################################################################################################################################################################################
    # create the column to identify whether keyword at middle
    df = read_avro(spark,  f"{hdfs_working_path}KeyOnlyData")
    mid = MiddleKey(keys)
    withMiddleFlag = df.withColumn("middle_key", mid.middleUDF(df.gram_type, df.token_gram))
    print("==========MARK THE NGRAM WITH KEYWORDS MIDDLE ON IT==========")
    write_avro(withMiddleFlag, f"{hdfs_working_path}MiddleFlagData")
    remove_from_hdfs(f"{hdfs_working_path}KeyOnlyData")
    print("remove keyword only and save with center flag")
    ###################################################################################################################################################################################
    # Frequency statistics
    # language pattern frequency statistics
    df = read_avro(spark,  f"{hdfs_working_path}MiddleFlagData")
    langFreq = df.groupBy("lang_gram").count().withColumnRenamed("count", "lang_gram_f")
    write_avro(langFreq, f"{hdfs_working_path}langFreq")
    langFreq = read_avro(spark, f"{hdfs_working_path}langFreq")
    mergeLangF = df.join(langFreq, df.lang_gram == langFreq.lang_gram).drop(langFreq.lang_gram)
    print("==========DF WITH LANG GRAM FREQUENCY==========")
    write_avro(mergeLangF, f"{hdfs_working_path}MergelangFreq")
    remove_from_hdfs(f"{hdfs_working_path}langFreq")
    ###################################################################################################################################################################################
    # token pattern frequency statistics
    df = read_avro(spark, f"{hdfs_working_path}MergelangFreq")
    tokenFreq = df.groupBy("token_gram").count().withColumnRenamed("count", "token_gram_f")
    write_avro(tokenFreq, f"{hdfs_working_path}tokenFreq")
    tokenFreq = read_avro(spark, f"{hdfs_working_path}tokenFreq")
    mergeTokenF = df.join(tokenFreq, df.token_gram == tokenFreq.token_gram).drop(tokenFreq.token_gram)
    print("==========DF WITH TOKEN FREQUENCY==========")
    write_avro(mergeTokenF, f"{hdfs_working_path}MergeTokenFreq")
    remove_from_hdfs(f"{hdfs_working_path}tokenFreq")
    remove_from_hdfs(f"{hdfs_working_path}MergelangFreq")
    ###################################################################################################################################################################################
    # tag pattern frequency statistics
    df = read_avro(spark, f"{hdfs_working_path}MergeTokenFreq")
    tagGramFreq = df.groupBy("tag_gram").count().withColumnRenamed("count", "tag_gram_f")
    write_avro(tagGramFreq, f"{hdfs_working_path}WithTagGramFreq")
    tf = read_avro(spark, f"{hdfs_working_path}WithTagGramFreq")
    withtagF = df.join(tf, df.tag_gram == tf.tag_gram).drop(tf.tag_gram)
    print("==========DF WITH TAG GRAM FREQUENCY==========")
    write_avro(withtagF, f"{hdfs_working_path}WithTagFreq")
    remove_from_hdfs(f"{hdfs_working_path}WithTagGramFreq")
    remove_from_hdfs(f"{hdfs_working_path}MergeTokenFreq")
    ###################################################################################################################################################################################
    # normalize into two table before store into hive
    # gram table & original data
    df = read_avro(spark, f"{hdfs_working_path}WithTagFreq")
    df =df.repartition(800)

    print("if disk block error happen, need increase the repartition value")

    gram_table = \
        df.select("sentence_id", "pos", "token_gram", "token_gram_f", \
                  "tag_gram", "tag_gram_f", "lang_gram", "lang_gram_f", \
                  "gram_type", "middle_key", "language")

    gram_table = gram_table.withColumn("uniqueID", monotonically_increasing_id())

    print("==========GETTING GRAM TABLE==========")
    write_avro(gram_table, f"{hdfs_working_path}gram_table")

    originalSentence  = df.dropDuplicates((['sentence_id'])) \
        .select("sentence_id", "original", "tag", \
                "language", "token", "chinese_tag", "language_pattern")

    originalSentence = originalSentence.withColumn("uniqueID", monotonically_increasing_id())

    print("==========GETTING SOURCE TABLE==========")
    write_avro(originalSentence, f"{hdfs_working_path}source_data")
    ###################################################################################################################################################################################
    # save into hive table
    spark.sql("drop database if exists social_media_db cascade;")
    spark.sql("create database social_media_db;")


    original = read_avro(spark, f"{hdfs_working_path}source_data")
    gram_table = read_avro(spark, f"{hdfs_working_path}gram_table")


    original.write\
    .format("orc") \
    .mode("overwrite") \
    .saveAsTable("social_media_db.source_data")


    gram_table.write\
    .format("orc") \
    .mode("overwrite") \
    .saveAsTable("social_media_db.gram_table")
    ###################################################################################################################################################################################
    # analytics : Top 10 Common Language Pattern
    spark.sql("use social_media_db;")

    Q="""
    select
        distinct lang_gram,
        lang_gram_f
    from
        gram_table
    order by
        lang_gram_f DESC limit 10;
    """

    SMLang = spark.sql(Q).toPandas()

    fig, ax = plt.subplots(1,1,figsize=(15,5))

    x = SMLang['lang_gram'][:10].tolist()
    y = SMLang['lang_gram_f'][:10].tolist()
    ax.barh(x,y)
    ax.set_title('Top 10 Most Common Language Patterns in Social Media Data')
    for index, value in enumerate(y):
        ax.text(value, index,
                 str(value))
    ax.invert_yaxis()

    import shutil
    try:
        shutil.rmtree("visualization")
    except:
        pass
    
    plt.savefig('visualization/top10LangPattern', dpi=300, bbox_inches='tight')
    print("Top 10 language pattern is ready")

    ###################################################################################################################################################################################
    # analytics : Top 10 Common Pos Pattern

    Q="""
    select
        distinct tag_gram,
        tag_gram_f
    from
        gram_table
    order by
        tag_gram_f DESC limit 10;
    """

    SMPos = spark.sql(Q).toPandas()



    fig, ax = plt.subplots(1,1,figsize=(15,5))

    x = SMPos['tag_gram'][:10].tolist()
    y = SMPos['tag_gram_f'][:10].tolist()
    ax.barh(x,y)
    ax.set_title('Top 10 Most Common POS Patterns in Social Media Data')
    for index, value in enumerate(y):
        ax.text(value, index,
                 str(value))
    ax.invert_yaxis()

    plt.savefig('visualization/top10PosPattern', dpi=300, bbox_inches='tight')
    print("Top 10 pos tag pattern is ready")

    ###################################################################################################################################################################################
    # analytics : Top 10 Common token Pattern

    Q="""
    select
        distinct token_gram,
        token_gram_f
    from
        gram_table
    order by
        token_gram_f DESC limit 10;
    """

    SMtok = spark.sql(Q).toPandas()

    fig, ax = plt.subplots(1,1,figsize=(15,5))

    x = SMtok['token_gram'][:10].tolist()
    y = SMtok['token_gram_f'][:10].tolist()
    ax.barh(x,y)
    ax.set_title('Top 10 Most Common Token Patterns in Social Media Data')
    for index, value in enumerate(y):
        ax.text(value, index,
                 str(value))
    ax.invert_yaxis()

    plt.savefig('visualization/top10TokenPattern', dpi=300, bbox_inches='tight')
    print("Top 10 token pattern is ready")


