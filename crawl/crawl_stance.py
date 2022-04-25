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

fileName = './stance/face_masks_test.csv'
with open(fileName, 'r', encoding='utf-8') as f:
    data = f.readlines()
data = [i.split(',')[0] for i in data[1:]]

for id in range(int(len(data)/100)):
    tweets = client.get_tweets(ids=data[id*100:(id+1)*100])
    for tweet in tweets.data:
        print(tweet)
if (id+1)*100 < len(data):
    tweets = client.get_tweets(data[(id+1)*100:])
    for tweet in tweets.data:
        print(tweet)
# for tweet in tweets.errors:
#     print(tweet)
# import pickle
# with open('hate_test.pickle', 'wb') as handle:
#     pickle.dump(tweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
# #
# with open('hate_test.pickle', 'rb') as handle:
#     b = pickle.load(handle)
