# app/text_normalizer.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re, unicodedata
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk

# Asegura recursos NLTK en runtime
for pkg, locator in [("punkt", "tokenizers/punkt"), ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(locator)
    except LookupError:
        nltk.download(pkg, quiet=True)

stemmer = SnowballStemmer("spanish")
_spanish_stopwords = set(stopwords.words("spanish"))

def replace_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")

def replace_numbers_with_text(text: str) -> str:
    num_map = {'0':'cero','1':'uno','2':'dos','3':'tres','4':'cuatro','5':'cinco','6':'seis','7':'siete','8':'ocho','9':'nueve'}
    return re.sub(r"\d", lambda m: num_map[m.group()], text)

def basic_normalize(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = replace_accents(text)
    text = replace_numbers_with_text(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_text(text: str) -> list[str]:
    if not text:
        return []
    return word_tokenize(text, language="spanish")

def stem_tokens(tokens: list[str], remove_sw: bool = True) -> list[str]:
    if remove_sw:
        tokens = [t for t in tokens if t not in _spanish_stopwords]
    return [stemmer.stem(t) for t in tokens if t.strip()]

def process_text_once(text: str, remove_stopwords: bool = True) -> str:
    basic = basic_normalize(text)
    toks = tokenize_text(basic)
    stems = stem_tokens(toks, remove_sw=remove_stopwords)
    return " ".join(stems)

class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(X).fillna("").astype(str)
        return s.apply(lambda t: process_text_once(t, remove_stopwords=self.remove_stopwords))
