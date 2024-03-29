# Ukuxhumana

"Ukuxhumana" means "Communicate" in Zulu. This project is aimed at exploring ideas for using Neural Machine Translation for low-resource languages - right now, specifically for the official languages of South Africa, but we are looking for collaborators across the continent to work together with us for the other languages

# Mission

- Provide a centralized repository for known datasets for African NMT and other NLP applications.
- Provide pretrained state-of-the-art models for African languages.
- Decrease the barrier to doing NMT research for African languages by providing code and data and models.
- Spur collaboration across the continent to work on these problems together.

# Data

## Parallel Corpuses

Our parallel corpuses are from [the Autshumato project](https://biblio.ugent.be/publication/1851705/file/6736544#page=39). The datasets contain data that was translated by professional translators, data that was sourced as translated file pairs from translators and data obtained from Government websites and documents. We also performed extra cleaning on the corpuses, which is described [here](https://github.com/LauraMartinus/ukuxhumana/blob/master/clean/README.md)


## Monolingual Corpuses

Our monolingual corpuses are from a variety of sources. We've used the monolingual corpuses for use in the training of fastText embeddings, which are also used in Unsupervised NMT.

### Zulu

- [Leipzig Zulu 100K Corpus](http://corpora.uni-leipzig.de/en?corpusId=zul_mixed_2016)
- [NCHLT isiZulu Text Corpora](https://rma.nwu.ac.za/index.php/isizulu-nchlt-text-corpora.html) cleaned by [Bernhard Duvenhage](https://github.com/praekelt/feersum-lid-shared-task)
### English

- WMT 2014

## Known Corpuses

We keep a list of known corpuses for African languages [here](https://github.com/LauraMartinus/ukuxhumana/blob/master/KNOWN_CORPUSES.md). Please consider contributing a link to your corpus :) 


# Models
Currently, two main architectures are used throughout this project, namely Convolutional Sequence to Sequence by [Gehring et. al. (2017)](https://arxiv.org/abs/1705.03122) and Transformer by [Vaswani et. al (2017)](https://arxiv.org/abs/1706.03762). [Fairseq(-py)](https://github.com/pytorch/fairseq) and Tensor2Tensor were used in modeling these techniques respectively. For each language, a model was trained using [byte-pair encoding](https://arxiv.org/abs/1508.07909) (BPE) for tokenisation. The learning rate was set to 0.25 and dropout to 0.2. Beam search with a width of 5 was used in decoding the test data.

The original [Tensor2Tensor implementation](https://github.com/tensorflow/tensor2tensor) of Transformer was used. The learning rate was set to 0.4, with a batch size of 1024, and a learning rate warm-up of 45000 steps. Tokenisation was done using [WordPiece](https://github.com/google/sentencepiece). Beam search with width 4 was used for decoding.


# Results
Results are given in BLEU.
## Baseline 
### English -> Language
| Model | Setswana | isiZulu* | Northern Sotho | Xitsonga | Afrikaans |
| ------- | ------- | ------- | ------- | ------- | ------- |
| Google Translate       |        | 7.55 |       |       | 41.181 |
| Convolutional Seq2Seq (clean)  | 24.18  | 0.28 | 7.41 | 36.96 | 16.17 |
| Convolutional Seq2Seq (best BPE) | 26.36 (40k)  | 1.79 (4k) | 12.18 (4k) | 37.45 (20k) | 25.04 (4k) |
| Transformer (uncased)  | 33.53  | 3.33 | 24.16 (4k) | 49.74 (20k) | 35.26 (4k) |
| Transformer (cased)    | 33.12  | 3.16 (4k) | 23.77 (4k) | 49.30 (20k) | 34.81 (4k) |
| [Unsupervised MT (60K BPE)](https://github.com/facebookresearch/UnsupervisedMT)    |   | 4.45 |  |  |  |


\* Zulu data requires cleaning. Translations often contain more information than in original sentence, leading to poor BLEU scores.

# Autshumato Machine Translation Benchmark 

| Model | Afrikaans | isiZulu | Northern Sotho | Setswana | Xitsonga |
| ------- | ------- | ------- | ------- | ------- |  ------- |
| Convolutional Seq2Seq | 12.30 | 0.52 | 7.41 | 10.31 | 10.73 |
| Transformer | 20.60 | 1.34 | 10.94 | 15.60 | 17.98 |



# Publications & Citations

[Benchmarking Neural Machine Translation for Southern African Languages](https://arxiv.org/pdf/1906.10511.pdf)

[A Focus on Neural Machine Translation for African Languages](https://arxiv.org/pdf/1906.05685.pdf)

[Towards Neural Machine Translation for African Languages](https://arxiv.org/abs/1811.05467)
