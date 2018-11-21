import codecs

with codecs.open("../clean/en_tn/entn_parallel.train.tn", 'r','utf-8') as in_f:
    data = in_f.readlines()
out_f = codecs.open("prepared_data/tn_train",'w','utf-8')

for line in data:
    line+= "<EOS>"
    out_f.writelines(line+ '\n')

out_f.close()

with codecs.open("../clean/en_tn/entn_parallel.dev.tn", 'r','utf-8') as in_f:
    data = in_f.readlines()
out_f = codecs.open("prepared_data/tn_dev",'w','utf-8')

for line in data:
    line+= "<EOS>"
    out_f.writelines(line+ '\n')

out_f.close()

with codecs.open("../clean/en_tn/entn_parallel.test.tn", 'r','utf-8') as in_f:
    data = in_f.readlines()
out_f = codecs.open("prepared_data/tn_test",'w','utf-8')

for line in data:
    line+= "<EOS>"
    out_f.writelines(line+ '\n')

out_f.close()