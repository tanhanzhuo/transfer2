import tweepy
import os
# consumer_key=os.environ.get("BEARER_TOKEN")
# consumer_secret=os.environ.get("BEARER_TOKEN")
# access_token=os.environ.get("BEARER_TOKEN")
# access_token_secret=os.environ.get("BEARER_TOKEN")
bearer_token = os.environ.get("BEARER_TOKEN")
client = tweepy.Client( bearer_token=bearer_token,
                        # consumer_key=consumer_key,
                        # consumer_secret=consumer_secret,
                        # access_token=access_token,
                        # access_token_secret=access_token_secret,
                        return_type=dict)
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
print(len(test_id))
total = 0
for id in range(int(len(test_id)/100)):
    tweets = client.get_tweets(ids=test_id[id*100:(id+1)*100])
    for tweet in tweets['data']:
        # print(tweet['text'])
        total+=1
tweets = client.get_tweets(test_id[(id+1)*100:])
# tweets = client.get_tweets(ids=test_id[:100])
for tweet in tweets['data']:
    # print(tweet['text'])
    total += 1
print(total)
# for tweet in tweets.errors:
#     print(tweet)

# data = ['696013355715272704','710578503041699840','1290755919404568577','1282319581068042242']
# tweets = client.get_tweets(ids=data)
# print(tweets)
# print(tweets['data'])
# print(tweets.data[0])
# for tweet in tweets.data:
#     print(tweet['Tweet id'])

# import pickle
# with open('hate_test.pickle', 'wb') as handle:
#     pickle.dump(tweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
# #
# with open('hate_test.pickle', 'rb') as handle:
#     b = pickle.load(handle)
