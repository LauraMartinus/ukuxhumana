# This is using sentence piece. Tensor2Tensor and sentence piece do not get along right now. Some weird encoding issue

import os
import sys
import sentencepiece as spm

translations = [
    "zu",
    "af",
    "nso",
    "ts",
    "tn"
]

root_folder = 'clean'
bpe_folder = 'bpe'

if len(sys.argv) > 1:
    vocab_size = int(sys.argv[1])
else:
    vocab_size = 8000 #default

print("vocab size: %s" % (vocab_size))

def encode_and_save_file(inputName, outputName, sp):
    print("input file: %s" % (inputName))
    print("output file: %s" % (outputName))

    with open(inputName,'r') as f:
        data = f.readlines()
    with open(outputName,"w") as f:
        for x in data:
            pieces = sp.EncodeAsPieces(x)
            line = " ".join(pieces)
            f.write(line+'\n')
    print("done")


for t in translations:
    lang_folder = os.path.join(root_folder, "en_%s" % (t,))
    destination_folder = os.path.join(bpe_folder, "en_%s" % (t,))
    L1_train = os.path.join(lang_folder, "en%s_parallel.train.en" % (t,))
    L2_train = os.path.join(lang_folder, "en%s_parallel.train.%s" % (t,t,))

    L1_output_train = os.path.join(destination_folder, "en%s_parallel.%d.train.en" % (t,vocab_size))
    L2_output_train = os.path.join(destination_folder, "en%s_parallel.%d.train.%s" % (t,vocab_size, t))

    L1_dev = os.path.join(lang_folder, "en%s_parallel.dev.en" % (t,))
    L2_dev = os.path.join(lang_folder, "en%s_parallel.dev.%s" % (t,t,))
    L1_output_dev = os.path.join(destination_folder, "en%s_parallel.%d.dev.en" % (t,vocab_size))
    L2_output_dev = os.path.join(destination_folder, "en%s_parallel.%d.dev.%s" % (t,vocab_size, t))


    L1_test = os.path.join(lang_folder, "en%s_parallel.test.en" % (t,))
    L2_test = os.path.join(lang_folder, "en%s_parallel.test.%s" % (t,t,))
    L1_output_test = os.path.join(destination_folder, "en%s_parallel.%d.test.en" % (t,vocab_size))
    L2_output_test = os.path.join(destination_folder, "en%s_parallel.%d.test.%s" % (t,vocab_size, t))

    files = [
        (L1_train, L1_output_train),
        (L2_train, L2_output_train),
        (L1_dev, L1_output_dev),
        (L2_dev, L2_output_dev),
        (L1_test, L1_output_test),
        (L2_test, L2_output_test),
    ]

    # Train the sentence piece trainer
    model_prefix = 'bpe/en_%s/bpe.%d' % (t, vocab_size,)
    spm.SentencePieceTrainer.Train('--input=%s,%s --model_prefix=%s --vocab_size=%d --character_coverage=1.0 --model_type=bpe' % 
                                    (L1_train, L2_train, model_prefix, vocab_size))
 
    sp = spm.SentencePieceProcessor()
    sp.Load("%s.model" % (model_prefix,))

    # Convert the files
    for x, y in files:
        encode_and_save_file(x, y, sp)

    # Clean the vocab for t2t (it doesn't like the probabilities)
    with open("%s.vocab" % (model_prefix), "r") as fin:
        with open("%s.tokens.vocab" % (model_prefix), "w") as fout:
            first = True
            for l in fin:
                if first == False:
                     fout.write("\n")
                fout.write(l.split("\t")[0])
                first = False