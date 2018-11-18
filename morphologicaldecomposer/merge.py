with open("outt", 'r') as in_f:
    data = in_f.readlines()
out_f = open("merge",'w')

merged = ''
# data = data[:20]
# print(data)
for line in data:
    if '< E O S >' in line: 
        out_f.writelines(merged+ '\n')
        merged = ''
    else:
        merged += line[0:-1]
        merged += ' '
    
    # if ';' in line:
    #     out_f.writelines(merged+ '\n')
    #     merged = ''
    # else: 
    #     if ',' in line:
    #         out_f.writelines(merged+ '\n')
    #         merged = ''
    #     else:
    #         if '.' in line:
    #             out_f.writelines(merged+ '\n')
    #             merged = ''