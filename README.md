# Ukuxhumana

"Ukuxhumana" means "Communicate" in Zulu. This project is aimed at exploring ideas for using Neural Machine Translation for low-resource languages - specifically for the official languages of South Africa.

# Data

## Parallel Corpuses

Our parallel corpuses are from [the Autshumato project](https://biblio.ugent.be/publication/1851705/file/6736544#page=39). The datasets contain data that was translated by professional translators, data that was sourced as translated file pairs from translators and data obtained from Government websites and documents

# Models
Two main architectures are used throughout this project, namely Convolutional Sequence to Sequence by Gehring et. al. and Transformer by Vaswani et. al. Fairseq(-py) and Tensor2Tensor were used in modeling these techniques respectively.

# Results
Results are given in BLEU.
## Baseline 
### English -> Language
| Model | Setswana | isiZulu | Northern Sotho | Xitsonga | Afrikaans |
| ------- | ------- |------- |------- |------- |------- |
| Convolutional Seq2Seq  | 27.77 (24.18)  | 0.62 (0.28) | 15.35 (7.41) | 36.96 | 16.17 |
| Convolutional Seq2Seq (40K BPE) |  23.83 | 1.44 | 4.89 | 34.28 | 21.06 |
| Convolutional Seq2Seq (8K BPE) |   | 2.19 | 15.45 |  | 26.78 |
| Transformer (uncased)  | 33.53  | 4.55 | 29.23 | 47.37 | 35.26 |
| Transformer (cased)    | 33.12  | 4.45 | 28.71 | 46.95 | 34.81 |
| Transformer (40k BPE) (uncased)  |  | 4.29 |  |  |  |
| Transformer (40k BPE) (cased)    |   | 4.14 |  |  |  |
| Transformer (8k BPE) (uncased)  |  |  |  |  |  |
| Transformer (8k BPE) (cased)    |   |  |  |  |  |

### Language -> English
| Model | Setswana | isiZulu | Northern Sotho | Xitsonga | Afrikaans |
| ------- | ------- |------- |------- |------- |------- |
| Convolutional Seq2Seq  |   |  |  |  |  |
| Transformer (uncased)  |   |  |  |  |  |
| Transformer (cased)    |   |  |  |  |  |
