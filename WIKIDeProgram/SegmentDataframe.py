#---------------------------------------------------------------------------------------------   
# count the word, after that sort by most frequent
# What is the difference between map and flatmap???
# map => [[x],[x],[x]]
# flatmap => [x, x, x]
def to_word_arr(x):
    if x.Comment != None and x.Comment!= "":
        return x.Comment.split()
    else:
        return []
#---------------------------------------------------------------------------------------------    
# if same key, sum val
def sum_value_if_same_key(K1_v1, K1_v2):
    return K1_v1+K1_v2    
#---------------------------------------------------------------------------------------------   
# create 10 segment on the data
def get_count(dataframe):
    return dataframe.count()

def create_index_range(total, segmentQty):
    rowQty = get_each_segment_row_qty(total, segmentQty)
    arr = []
    increment = rowQty-1
    start = 0
    while(start < total):
        arr.append((start, start + increment))
        start = start + increment + 1
    return arr

def get_each_segment_row_qty(total, segmentQty):
    return int(total / segmentQty)
    

def get_segments(dataframe, segmentQty):
    segments = []
    index_range = create_index_range(dataframe.count(), segmentQty)
    for aRange in index_range:
        segment = dataframe.filter((dataframe.id >= aRange[0]) & (dataframe.id <= aRange[1]))
        segments.append(segment)
    return segments

#---------------------------------------------------------------------------------------------
def get_df_range_from_id(id_col_name, df, from_num, to_num):
    return df.filter((df['idx'] <= to_num) & (df['idx'] >= from_num))
#---------------------------------------------------------------------------------------------
def to_python_array(df, col_name):
    return df.select(col_name).rdd.flatMap(lambda x: x).collect()
#---------------------------------------------------------------------------------------------