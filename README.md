# ukuxhumana

"Ukuxhumana" means "Communicate" in Zulu. This project is aimed at exploring ideas for using Neural Machine Translation for Low-resource languages - In particular, for the Bantu languages of Africa.

# Results
Results are given in BLEU.
## Baseline 

| Model | Setswana | isiZulu | Northern Sotho | Xitsonga | Afrikaans |
| ------- | ------- |------- |------- |------- |------- |
| Convolutional Seq2Seq  | 27.77  | 0.62 | 15.35 | 36.96 | 16.17 |
| Transformer (uncased)  | 33.53  | 4.55 | 29.23 |       |   |
| Transformer (cased)    | 33.12  | 4.45 | 28.71 |       |   |