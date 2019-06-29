import codecs

directory = "clean/en_tn/"
pref = "entn_parallel"
suf = ".tn"

en_transcriptions = [line.rstrip('\n') for line in codecs.open("data/autshumato_eval_set/english.txt", "r", "utf-8")]
t_transcriptions = [line.rstrip('\n') for line in codecs.open("data/autshumato_eval_set/setswana.txt", "r", "utf-8")]

eng_test = codecs.open(directory+pref+'.test.en','w','utf-8')
t_test = codecs.open(directory+pref+'.test'+suf,'w','utf-8')

for counter in range(len(en_transcriptions)):

    eng_test.write(u''.join((en_transcriptions[counter])))
    t_test.write(u''.join((t_transcriptions[counter])))

