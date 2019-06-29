import codecs

directory = "clean/en_nso/"
pref = "ennso_parallel"
suf = ".nso"

eng_dev = codecs.open(directory+pref+'.dev.en','w','utf-8')
t_dev = codecs.open(directory+pref+'.dev'+suf,'w','utf-8')
eng_train = codecs.open(directory+pref+'.train.en','w','utf-8')
t_train = codecs.open(directory+pref+'.train'+suf,'w','utf-8')

en_transcriptions = [line.rstrip('\n') for line in codecs.open(directory+pref+".en", "r", "utf-8")]
t_transcriptions = [line.rstrip('\n') for line in codecs.open(directory+pref+suf, "r", "utf-8")]
for counter in range(len(en_transcriptions)):
    if counter < 3000:
        eng_dev.write(u''.join((en_transcriptions[counter])))
        t_dev.write(u''.join((t_transcriptions[counter])))
    else:
        eng_train.write(u''.join((en_transcriptions[counter])))
        t_train.write(u''.join((t_transcriptions[counter])))

