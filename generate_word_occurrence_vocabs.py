import sys
from nltk import ngrams, FreqDist
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

# usage: python generate_word_occurrence_vocabs.py <path/to/source> <path/to/output>

with open(sys.argv[1],"r") as f:
    content = f.read()

t = TreebankWordTokenizer()
data = t.tokenize(content)
fq = FreqDist(data)

print("Number of individual tokens: %d" %(fq.N()))
print("Number of unique tokens: %d" %(fq.B()))

with open(sys.argv[2],"w") as f:
    for x in fq.keys():
        f.write(x + "," + str(fq[x]) +"\n")
