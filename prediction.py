
import fasttext
from fasttext import load_model

from nltk.tokenize import MWETokenizer, word_tokenize, RegexpTokenizer
import re
import nltk
import unicodedata

normalizer = {'òa': 'oà',
              'óa': 'oá',
              'ỏa': 'oả',
              'õa': 'oã',
              'ọa': 'oạ',
              'òe': 'oè',
              'óe': 'oé',
              'ỏe': 'oẻ',
              'õe': 'oẽ',
              'ọe': 'oẹ',
              'ùy': 'uỳ',
              'úy': 'uý',
              'ủy': 'uỷ',
              'ũy': 'uỹ',
              'ụy': 'uỵ',
              'Ủy': 'Uỷ'}

multiple_punctuation_pattern = re.compile(r"([\"\.\?\!\,\:\;\-])(?:[\"\.\?\!\,\:\;\-]){1,}")
word_tokenizer = MWETokenizer(separator='')
multiple_emoji_pattern = re.compile(u"(["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\u00a9"
        u"\u00ae"
        u"\u2000-\u3300"
        "]){1,}", flags= re.UNICODE )


def normalize_text(text):
  for absurd, normal in normalizer.items():
    text = text.replace(absurd, normal)
  return text

def preprocess(text):
  text = unicodedata.normalize("NFC", text)
  text = multiple_emoji_pattern.sub(r"\g<1> ", text)
  text = multiple_punctuation_pattern.sub(r" \g<1> ", text)
  text = normalize_text(text.lower())
  text = re.sub('(free|ship|toàn quốc|nhập mã)', '',text)
  text = re.sub('(\()[\s\w]*(\))', '', text)
  text = re.sub('(\[)[\s\w\%\d]+(\])', '', text)
  text = re.sub('[^\w]+', ' ', text)
  text = re.sub('(\d+\w+)', '', text)
  text = re.sub('\d+', '', text)
  text = word_tokenizer.tokenize(word_tokenize(text))
  return ' '.join(text)

class Prediction:
    def __init__(self):
        # Load fasttext model
        self.model = load_model('./model_sell_detection.bin')

    def process(self, string_line):
        # pre process sent
        preprocessed_string = preprocess(string_line)
        # return prediction
        return self.model.predict(preprocessed_string)[0][0]