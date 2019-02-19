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
Two main architectures are used throughout this project, namely Convolutional Sequence to Sequence by Gehring et. al. and Transformer by Vaswani et. al. Fairseq(-py) and Tensor2Tensor were used in modeling these techniques respectively.

# Results
Results are given in BLEU.
## Baseline 
### English -> Language
| Model | Setswana | isiZulu* | Northern Sotho | Xitsonga | Afrikaans |
| ------- | ------- |------- |------- |------- |------- |
| Convolutional Seq2Seq (clean)  | 24.18  | 0.28 | 7.41 | 36.96 | 16.17 |
| Convolutional Seq2Seq (best BPE) |  | 1.79 (4k) | 12.18 (4k) |  | 25.04 (4k) |
| Transformer (uncased)  | 33.53  | 4.55 | 29.23 | 47.37 | 35.26 |
| Transformer (cased)    | 33.12  | 4.45 | 28.71 | 46.95 | 34.81 |
| [Unsupervised MT (60K BPE)](https://github.com/facebookresearch/UnsupervisedMT)    |   | 4.45 |  |  |  |

### Language -> English
| Model | Setswana | isiZulu* | Northern Sotho | Xitsonga | Afrikaans |
| ------- | ------- |------- |------- |------- |------- |
| Google Translate       |        | 7.55 |       |       | 41.181 |
| Convolutional Seq2Seq (clean)  |   | 0.95 | 5.80 | 39.19 | 25.99 |
| Convolutional Seq2Seq (morfessor)  |   | 2.38 |  |  |  |
| Convolutional Seq2Seq (bpe & morfessor)  |   | 5.06 |  |  |  |
| Convolutional Seq2Seq (bpe)  |   | 5.13 (4k) | 11.98 (8k) | 36.11 (8k) | 26.76 (8k) |
| Transformer (uncased)  |   |  |  |  |  |
| Transformer (cased)    |   |  |  |  |  |

\* Zulu data requires cleaning. Translations often contain more information than in original sentence, leading to poor BLEU scores.

# Publications & Citations

[Towards Neural Machine Translation for African Languages](https://arxiv.org/abs/1811.05467)

Please cite:
```
@article{abbott2018towards,
  title={Towards Neural Machine Translation for African Languages},
  author={Abbott, Jade Z and Martinus, Laura},
  journal={arXiv preprint arXiv:1811.05467},
  year={2018}
}
```
