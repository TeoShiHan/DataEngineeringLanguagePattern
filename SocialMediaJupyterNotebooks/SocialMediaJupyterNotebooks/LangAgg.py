def get_all_type_of_languages_detected(df, col_name):
    return df.select(df[col_name]).distinct().rdd.flatMap(lambda x : x).collect()

def map_language_with_its_dataframe(lang_arr, df, col_name):
    temp_map = {}
    for lang in lang_arr:
        temp_map[lang] = df.filter( df[col_name] == lang)
    return temp_map

class LanguageAggregor:
    def __init__(self, df, col_name):
        self.col_name = col_name
        self.df = df
        self.all_language = get_all_type_of_languages_detected(df, self.col_name)
        self.lang_df_map = map_language_with_its_dataframe(self.all_language, self.df, self.col_name)

    def print_available_lang(self):
        print(self.all_language)
        
    def get_lang_dataframe(self, lang):
        return self.lang_df_map[lang]
    
    def show_sample(self, sample_size):
        for lang in self.all_language:
            print("This is dataframe of lang ", lang)
            self.lang_df_map[lang].show(sample_size, False)