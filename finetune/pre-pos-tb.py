def convert_data(fileName, saveName):
    f = open(fileName, 'r')
    f_w = open(saveName, 'a')

    lines = f.readlines()
    for line in lines:
        if line[0].isdigit():
            line_split = line.split()
            f_w.write(line_split[1].strip() + '\t' + line_split[3].strip() + '\n')
        if line == '\n':
            f_w.write('\n')
    f.close()
    f_w.close()

convert_data('./data/pos-tb/dev_ori', './data/pos-tb/dev')