with open("mydata", 'r') as in_f:
    data = in_f.readlines()
out_f = open("eos",'w')

for line in data:
    line+= "<EOS>"
    out_f.writelines(line+ '\n')