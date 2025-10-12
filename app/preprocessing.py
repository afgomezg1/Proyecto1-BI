import re, unicodedata
import pandas as pd
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Inicialización de NLTK (solo 1 vez)
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

spanish_stopwords = set(stopwords.words("spanish"))
stemmer = SnowballStemmer("spanish")

def replace_accents(text: str) -> str:
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')

def replace_numbers_with_text(text: str) -> str:
    num_map = {'0':'cero','1':'uno','2':'dos','3':'tres','4':'cuatro','5':'cinco','6':'seis','7':'siete','8':'ocho','9':'nueve'}
    return re.sub(r'\d', lambda m: num_map[m.group()], text)

def basic_normalize(text: str) -> str:
    if text is None: return ""
    text = str(text).lower().strip()
    text = replace_accents(text)
    text = replace_numbers_with_text(text)
    text = re.sub(r'[^a-z\s]', ' ', text, flags=re.UNICODE)  # solo letras y espacios
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text: str) -> List[str]:
    if not text: return []
    return word_tokenize(text, language="spanish")

def stem_tokens(tokens: List[str], remove_sw: bool = True) -> List[str]:
    if remove_sw:
        tokens = [t for t in tokens if t not in spanish_stopwords]
    return [stemmer.stem(t) for t in tokens if t.strip()]

def process_text_once(text: str, remove_stopwords: bool = True) -> str:
    basic = basic_normalize(text)
    toks  = tokenize_text(basic)
    stems = stem_tokens(toks, remove_sw=remove_stopwords)
    return " ".join(stems)

def preprocess_for_api(texts: List[str], remove_stopwords: bool = True) -> pd.Series:
    s = pd.Series(texts).fillna("").astype(str)
    return s.apply(lambda t: process_text_once(t, remove_stopwords=remove_stopwords)).astype(str)
