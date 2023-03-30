import fasttext
model = fasttext.load_model('data/model_fasttext.bin')
def predict_language(msg):
    pred = model.predict([msg])[0][0]
    pred = pred.replace('__label__', '')
    return pred
