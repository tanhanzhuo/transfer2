import tweepy
consumer_key="b0qF9CcEslZWdfTY6Gyfr6k7i"
consumer_secret="x8tKpNAUbeiCaTCFyIBmDvpXHsCD91ZOx0MHq4rU7ZN92yZI4k"
access_token="1517497475749539841-9LVF5opQbfkuUnreMz5v3PsyxZCID1"
access_token_secret="Ok2vxvbvJWS579a3fAgjkytr5iVoZMWwVUGMLr5FkhJcQ"
bearer_token ="AAAAAAAAAAAAAAAAAAAAAAV7bwEAAAAAhAiNNg3waFSs8qhlJQC1Q3V7mX0%3DuJztuSt5BI0tcjugQH0bEQdvkWwYxV8EXb1ErCZfpVTK6zw0pc"
client = tweepy.Client( bearer_token=bearer_token,
                        consumer_key=consumer_key,
                        consumer_secret=consumer_secret,
                        access_token=access_token,
                        access_token_secret=access_token_secret
                        )
import json
import numpy as np

fileName = './hate/dataset.json'
fileSP = './hate/post_id_divisions.json'

with open(fileName, 'r', encoding='utf-8') as f:
        data = json.load(f)
with open(fileSP, 'r', encoding='utf-8') as f:
    division = json.load(f)
print(fileName)

division_label = {}
for sp in division.keys():
    sp_id_all = division[sp]
    division_label[sp] = []
    for id in sp_id_all:
        if 'twitter' in id:
            ann_final = [0,0,0]
            annotations = data[id]['annotators']
            for ann_one in annotations:
                if ann_one['label'] == 'hatespeech':
                    ann_final[0] += 1
                elif ann_one['label'] == 'normal':
                    ann_final[1] += 1
                elif ann_one['label'] == 'offensive':
                    ann_final[2] += 1
                else:
                    print('error:' + id)
            if ann_final == [1,1,1]:
                continue
            label = np.argmax(ann_one)
            division_label[sp].append([id.split('_')[0], label])
test_id = [i[0] for i in division_label['test']]
test_id=['696013355715272704','710578503041699840','1290755919404568577','1282319581068042242']
tweets = client.get_tweets(test_id[:100])
print(tweets)
import pickle
with open('hate_test.pickle', 'wb') as handle:
    pickle.dump(tweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('hate_test.pickle', 'rb') as handle:
#     b = pickle.load(handle)
