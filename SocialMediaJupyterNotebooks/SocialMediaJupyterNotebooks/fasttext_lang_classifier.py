import fasttext
    
# Load model (loads when this library is being imported)
model = fasttext.load_model('/home/pc/Assignment/PretrainedModel/lid.176.bin')

# This is the function we use in UDF to predict the language of a given msg
def predict_language(msg):
    pred = model.predict([msg])[0][0]
    pred = pred[0].replace('__label__', '')
    return pred