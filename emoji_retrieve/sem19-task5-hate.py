from tqdm import tqdm
import re
import random
THRE = 1000
with open('sem19-task5-hate.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
keywords = []
for line in lines:
    line = line.strip().split('- ')[-1]
    keywords.append(line)
word_dic = {}
for one in keywords:
    word_dic[one] = 0

# filePath = '/work/data/twitter_ref.txt'
filePath = './data_sem19-task5-hate3.txt'
with open(filePath, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)
pattern = '|'.join(keywords)
data = []
for line in tqdm(lines):
    line_lower = line.lower()

    # results = re.search(pattern,line_lower)
    # if results:
    #     data.append(line)
    #     word_dic[results.group()]+=1
    for word in keywords:
        if word in line_lower:
            if word_dic[word] > THRE:
                continue
            data.append(line)
            word_dic[word]+=1
            break
print(word_dic)
with open('data_sem19-task5-hate.txt', 'w') as f:
    for one in data:
        f.write(one)