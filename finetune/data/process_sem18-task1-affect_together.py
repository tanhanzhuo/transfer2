import random
def write_txt(fileName, data):
    with open(fileName, 'w', encoding='utf-8') as f:
        for one in data:
            f.write(one)

for sp in ['train','dev','test']:
    data_tmp = []
    for EMO in ['anger', 'fear', 'joy', 'sadness']:
        with open('./sem18-task1-affect/'+EMO+'/'+sp,'r',encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line_sp = line.split('\t')
                line_sp[0] = line_sp[0].split(':')[0]
                line_sp[1] = line_sp[1].replace('\n',' ')
                line_new = line_sp[0] + '\t' + line_sp[1] + ' \n'
                data_tmp.append(line_new)
    write_txt('./sem18-task1-affect/'+sp,data_tmp)