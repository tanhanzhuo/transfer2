import tweepy
from tqdm import trange,tqdm
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

fileName = './wtwt/wtwt_ids.json'
with open(fileName, 'r', encoding='utf-8') as f:
    data = json.load(f)
data = [i['tweet_id'] for i in data]

data = data[25000:]

data_crawl = []
total = 0
for id in trange(int(len(data)/100)):
    tweets = client.get_tweets(ids=data[id*100:(id+1)*100])
    for tweet in tweets['data']:
        data_crawl.append(tweet)
        total+=1
if (id+1)*100 < len(data):
    tweets = client.get_tweets(data[(id+1)*100:])
    for tweet in tweets['data']:
        data_crawl.append(tweet)
        total += 1
print('total number crawled:' + str(total))

with open('wtwt.json', 'a') as f:
    for tweet in data_crawl:
        json.dump(tweet, f, ensure_ascii=False)
        f.write('\n')
# for tweet in tweets.errors:
#     print(tweet)
# import pickle
# with open('hate_test.pickle', 'wb') as handle:
#     pickle.dump(tweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
# #
# with open('hate_test.pickle', 'rb') as handle:
#     b = pickle.load(handle)
