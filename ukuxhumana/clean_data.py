import sys
import os
import codecs
import string

def analyseTrans():
    eng_file = codecs.open('clean_en.txt','w','utf-8')
    t_file = codecs.open('clean_ts.txt','w','utf-8') 
    
    global_trans_counts = {}
    en_transcriptions = [line.rstrip('\n') for line in codecs.open("data/en_ts/ents_parallel.en", "r", "utf-8")]
    t_transcriptions = [line.rstrip('\n') for line in codecs.open("data/en_ts/ents_parallel.ts", "r", "utf-8")]
    for counter, transcription in enumerate(en_transcriptions):
        if transcription not in global_trans_counts:
            global_trans_counts[transcription] = 1
            eng_file.write(u''.join((transcription)))
            t_file.write(u''.join((t_transcriptions[counter])))
        else:
            global_trans_counts[transcription] += 1
            
    eng_file.close()
    t_file.close()


if __name__ == "__main__":
    analyseTrans()