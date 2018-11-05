##Requirements: pip install fasttext
import fasttext
import codecs

# Skipgram model
# model = fasttext.skipgram('data/mono_nchlt/improved_afr.txt', 'fastText/mono_nchlt/afr', dim=300)
# model = fasttext.skipgram('data/mono_nchlt/improved_eng.txt', 'fastText/mono_nchlt/eng', dim=300)
# model = fasttext.skipgram('data/mono_nchlt/improved_nbl.txt', 'fastText/mono_nchlt/nbl', dim=300)
# model = fasttext.skipgram('data/mono_nchlt/improved_nso.txt', 'fastText/mono_nchlt/nso', dim=300)
# model = fasttext.skipgram('data/mono_nchlt/improved_sot.txt', 'fastText/mono_nchlt/sot', dim=300)
# model = fasttext.skipgram('data/mono_nchlt/improved_ssw.txt', 'fastText/mono_nchlt/ssw', dim=300)
# model = fasttext.skipgram('data/mono_nchlt/improved_tsn.txt', 'fastText/mono_nchlt/tsn', dim=300)
# model = fasttext.skipgram('data/mono_nchlt/improved_tso.txt', 'fastText/mono_nchlt/tso', dim=300)
# model = fasttext.skipgram('data/mono_nchlt/improved_ven.txt', 'fastText/mono_nchlt/ven', dim=300)
# model = fasttext.skipgram('data/mono_nchlt/improved_xho.txt', 'fastText/mono_nchlt/xho', dim=300)
model = fasttext.skipgram('data/mono_nchlt/improved_zul.txt', 'fastText/mono_nchlt/zul', dim=300)
