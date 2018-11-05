##Requirements: pip install fasttext
import fasttext
import codecs

# Skipgram model
model = fasttext.skipgram('data/nchlt_zul/nchlt_zul.txt', 'fastText/nchlt_zul/model', dim=300)