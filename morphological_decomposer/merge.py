import codecs

with open("unmerged_data/afr_train.txt", 'r',encoding='utf8') as in_f:
    data = in_f.readlines()
out_f = codecs.open("decomposed_data/enaf_parallel.train.decomp.af",'w','utf-8')

merged = ''
for line in data:
    #if '< E O S >' in line: 
    if '<EOS>' in line:
        out_f.writelines(merged+ '\n')
        merged = ''
    else:
        merged += line[0:-1]
        merged += ' '

out_f.close()

with codecs.open("unmerged_data/afr_dev.txt", 'r','utf-8') as in_f:
    data = in_f.readlines()
out_f = codecs.open("decomposed_data/enaf_parallel.dev.decomp.af",'w','utf-8')

merged = ''
for line in data:
    if '<EOS>' in line: 
        out_f.writelines(merged+ '\n')
        merged = ''
    else:
        merged += line[0:-1]
        merged += ' '

out_f.close()

with open("unmerged_data/afr_test.txt", 'r',encoding='utf8') as in_f:
    data = in_f.readlines()
out_f = codecs.open("decomposed_data/enaf_parallel.test.decomp.af",'w','utf-8')

merged = ''
for line in data:
    if '<EOS>' in line:
        out_f.writelines(merged+ '\n')
        merged = ''
    else:
        merged += line[0:-1]
        merged += ' '

out_f.close()