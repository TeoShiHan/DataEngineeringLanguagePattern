import re
from functools import partial
from malaya.path import PATH_NGRAM, S3_PATH_NGRAM
from malaya.function import check_file
from malaya.dictionary import is_english, is_malay
from malaya.text.rules import rules_normalizer
from malaya.text.tatabahasa import stopword_tatabahasa
from malaya.spelling_correction.probability import Spell
from herpetologist import check_type
from typing import List
from pyspark import SparkFiles


class Spylls(Spell):
    def __init__(self, dictionary):
        self._dictionary = dictionary

    @check_type
    def correct(self, word: str):
        if is_english(word):
            return word
        if is_malay(word):
            return word
        if word in stopword_tatabahasa:
            return word

        if word in rules_normalizer:
            return rules_normalizer[word]
        else:
            r = self.edit_candidates(word=word)[:1]
            if len(r):
                return r[0]
            else:
                return word
    @check_type
    def edit_candidates(self, word: str):
        return list(self._dictionary.suggest(word))

@check_type
def load(modelpath, model: str = 'libreoffice-pejam', **kwargs):
    try:
        from spylls.hunspell import Dictionary
    except BaseException:
        raise ModuleNotFoundError(
            'spylls not installed. Please install it and try again.'
        )

    model = model.lower()
    supported_models = ['libreoffice-pejam']
    if model not in supported_models:
        raise ValueError(
            f'model not supported, available models are {str(supported_models)}'
        )

    try:
        dictionary = Dictionary.from_zip(modelpath)
    except BaseException:
        raise Exception('failed to load spylls model, please try clear cache or rerun again.')
    return Spylls(dictionary=dictionary)