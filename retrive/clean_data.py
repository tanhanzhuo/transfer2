with open('data_extension.txt', 'r') as f:
    lines = f.readlines()
with open('data_extension_clean.txt', 'w') as f:
    for line in lines:
        tmp = line.replace('[RT]', '').replace('[USER]', '@USER').replace('[HTTP]', 'https://')
        f.write(tmp)