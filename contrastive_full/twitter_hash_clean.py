import re
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file',default='twitter_hash',type=str)
args = parser.parse_args()

HASH = re.compile(r"#\S+")
def read_data(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        data = set()
        lines = f.readlines()
        print('read lines')
        for line in tqdm(lines):
            line_clean = line.replace('[RT] ', '').replace('[USER]', '@USER').replace('[HTTP]', 'https').strip()
            line_no = line_clean.replace('@USER', '').replace('https', '').replace(' ','')
            if len(line_no) > 10:
                hash_tmp = HASH.findall(line_clean)
                for hash_one in hash_tmp:
                    line_no = line_no.replace(hash_one,'')
                if len(line_no) > 10:
                    data.add(line_clean)
    return data

def write_data(fileName):
    data = list(read_data(fileName+'.txt'))
    print('write lines')
    with open(fileName+'_clean.txt', 'w', encoding='utf-8') as f:
        for one in tqdm(data):
            f.write(one + ' \n')
write_data(args.file)