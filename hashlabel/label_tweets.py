from prettytable import PrettyTable
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--width',default=50,type=int)
parser.add_argument('--num',default=2014,type=int)
args = parser.parse_args()

with open('hash_pair.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip().split('\t')
        data.append(line)

with open('hash_comment.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    comment = {}
    for line in lines:
        line = line.strip().split('\t')
        comment[line[0]] = line[1]


with open('hash_sample_random10.json', 'r', encoding='utf-8') as f:
    hash_sample_random = json.load(f)

# NUM=10
# total = 0
# for idx_data in range(len(data)):
#     pair = data[idx_data]
#     if len(pair) == 3:
#         total += 1
#         continue
#     if len(pair) != 2:
#         print('!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!')
#     hash_one = pair[0]
#     hash_two = pair[1]
#     table = PrettyTable([str(idx_data),hash_one+': '+comment[hash_one],hash_two+': '+comment[hash_two]],max_width=args.width)
#     table.hrules=True
#     table.align = 'l'
#     for idx in range(NUM):
#         table.add_row([str(idx), hash_sample_random[hash_one][idx], hash_sample_random[hash_two][idx]])
#     print(table)
#     print('pls rate the similarity between hashtags and select from [0,1,2,3]')
#     print('0: In general these two hashtags cannot replace each other. They refer to different topics and not similar to each other')
#     print('1: In few scenario these two hashtags can replace each other in the tweets. A) they refer to the somewhat similar topics; and, '\
#           'B) one topic is a extended relative of another in the category tree (not parents, siblings or offspring)')
#     print('2: In lots of scenario these two hashtags can replace each other in the tweets (neglect the time/location restrictions): '\
#           'A) they refer to the very similar topics; and, B) one topic is a first-degree relative of another in the category tree (parents, siblings and offspring)')
#     print('3: In most scenario these two hashtags can replace each other in the tweets. They refer to the same topic')
#     rate = input('type your rate: ')
#     while rate not in ['0','1','2','3']:
#         print('wrong input! should be [0,1,2,3]')
#         rate = input('type your rate: ')
#     pair.append(rate)
#     print('***********************************************************************************************')
#     print('***********************************************************************************************')
#     total += 1
#     with open('hash_pair.txt', 'w', encoding='utf-8') as f:
#         for pair in data:
#             f.write('\t'.join(pair) + '\n')
#     if total > args.num:
#         print('DONE! Thank you!')
#         break


NUM=10
total = 0
for idx_data in range(len(data)):
    pair = data[idx_data]
    if len(pair) == 3:
        total += 1
        continue
    if len(pair) != 2:
        print('!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!')
    hash_one = pair[0]
    hash_two = pair[1]
    table = PrettyTable([str(idx_data),hash_one+': '+comment[hash_one],hash_two+': '+comment[hash_two]],max_width=args.width)
    table.hrules=True
    table.align = 'l'
    for idx in range(NUM):
        table.add_row([str(idx), hash_sample_random[hash_one][idx], hash_sample_random[hash_two][idx]])
    print(table)
    print('pls indicate how many percentage of tweets, these two hashtags can replace each other.')
    print('select from [0~99]')
    while 1:
        rate = input('type your rate: ')
        if rate.isdigit():
            rate = int(rate)
            if rate >=0 and rate<100:
                break
        print('wrong input! should be [0~99]')
    pair.append(str(rate))
    print('***********************************************************************************************')
    print('***********************************************************************************************')
    total += 1
    with open('hash_pair.txt', 'w', encoding='utf-8') as f:
        for pair in data:
            f.write('\t'.join(pair) + '\n')
    if total > args.num:
        print('DONE! Thank you!')
        break