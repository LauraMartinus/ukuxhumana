# ukuxhumana

"Ukuxhumana" means "Communicate" in Zulu. This project is aimed at exploring ideas for using Neural Machine Translation for Low-resource languages - In particular, for the Bantu languages of Africa.

# Install & Run

Runs on a single GPU:

`./run_t2t_train.sh`

# Results
## Baseline 
### Setswana
Convolutional Seq2Seq: 27.77 BLEU <br />
Transformer (uncased): 33.53 BLEU <br />
Transformer (cased): 33.12 BLEU <br />

### isiZulu
Convolutional Seq2Seq: 0.62 BLEU <br />
Transformer (uncased): 4.55 BLEU <br />
Transformer (cased): 4.45 BLEU <br />

### Northern Sotho
Convolutional Seq2Seq: 15.35 BLEU <br />
Transformer (uncased):  <br />
Transformer (cased): <br />
