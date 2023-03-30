bmSeq = 1
bcSeq = 1
langDF = {"bm" : [], "bc" :[]}

def get_chinese_df(df):
    return df.filter(df.text.rlike(u'[\u4E00-\u9FA5]'))
    
def get_non_chinese_df(df):
    return df.filter(df.text.rlike(u'[^\u4E00-\u9FA5]'))


def show_menu():
    t = """
    1. show another sample
    2. bm and bc both valid
    3. bm valid only
    4. bc valid only
    5. both not valid
    """
    
    print(t)

def get_input_and_action(dictdf, lang):
    print("FOR LANG ", lang)
    print("WITH CHINESE")
    chineseDF = get_chinese_df(dictdf[lang])
    print("WITHOUT CHINESE")
    nonChineseDF = get_non_chinese_df(dictdf[lang])
    chineseDF.sample(0.06).show(5, False)
    nonChineseDF.sample(0.06).show(5, False)
    show_menu()
    choice = input("enter your choice : ")
    choice = int(choice)
    
    if choice == 1:
        return True
    elif choice == 2:
        langDF["bm"].append(nonChineseDF)
        langDF["bc"].append(chineseDF)
    elif choice == 3:
        langDF["bm"].append(nonChineseDF)
    elif choice == 4:
        langDF["bc"].append(chineseDF)
    else:
        pass
    return False

        
        
def startCustomAggregation(dictDF, langArr):
    for lang in langArr:
        unresolved = True
        while(unresolved):
            unresolved = get_input_and_action(dictDF, lang)
    return langDF