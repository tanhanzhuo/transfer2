import re
import json
import emoji
HASH = re.compile(r"#\S+")

def find_emoji_hashtag(data, mode):
    if mode == 'emoji':
        total = 0
        for one in data:
            if len(emoji.distinct_emoji_list(one)) > 0:
                total += 1
        return total
    elif mode == 'hashtag':
        total = 0
        for one in data:
            if len(HASH.findall(one)) > 0:
                total += 1
        return total
    elif mode == 'emoji_hashtag':
        total = 0
        for one in data:
            if len(HASH.findall(one)) > 0 and len(emoji.distinct_emoji_list(one)) > 0:
                total += 1
        return total
    else:
        return None
###count
# for task in 'stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm'.split(','):
#     for sp in ['train','dev','test']:
#         data = []
#         with open('../transfer2/finetune/data/'+task+'/'+sp+'.json') as f:
#             for line in f:
#                 data.append(json.loads(line)['text'])
#         emo = find_emoji_hashtag(data,'emoji')
#         hash = find_emoji_hashtag(data, 'hashtag')
#         emo_hash = find_emoji_hashtag(data, 'emoji_hashtag')
#         print('task:{},split:{},total:{},emo:{},hash:{},emo_hash:{}'.format(task,sp,len(data),emo,hash,emo_hash))

###extract
def write_json(fileName,data):
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in data:
            json.dump(one, f)
            f.write('\n')

for task in 'sem-18,sem19-task5-hate,sem19-task6-offen'.split(','):
    for sp in ['train','dev','test']:
        data = []
        with open('../finetune/data/'+task+'/'+sp+'.json') as f:
            for line in f:
                one  = json.loads(line)
                hash_one = HASH.findall(one['text'])
                emoji_one = emoji.distinct_emoji_list(one['text'])
                if len(hash_one) > 0 and len(emoji_one) > 0:
                    one['hash'] = hash_one
                    one['emoji'] = emoji_one
                    data.append(one)
        write_json('../finetune/data/'+task+'/'+sp+'_emo_hash.json', data)