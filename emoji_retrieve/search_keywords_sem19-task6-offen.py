import tweepy
from pagination import Paginator
import os
emo_100={'đ ': 39968, 'đŖ': 41421, 'đ': 42538, 'ÂŽ': 43092, 'đļ': 43145, 'đđž': 43797, 'đ': 44462, 'đ': 46428, 'đĨ': 46615, 'âī¸': 46644, 'âšī¸': 46909, 'đ': 48557, 'ÂŠ': 49189, 'đ': 49252, 'đĨ°': 49539, 'đ': 55596, 'đ': 57845, 'đģ': 59191, 'đˇ': 60233, 'đĒ': 60579, 'â': 60885, 'đŖ': 60888, 'đēđ¸': 61952, 'â': 63647, 'âŧī¸': 63962, 'đ': 64140, 'đ': 64764, 'đ': 64790, 'đ': 64805, 'đˇ': 66930, 'đ': 67046, 'đ': 67355, 'đ¤': 71526, 'đ': 72717, 'đ': 72994, 'đ': 73519, 'đ': 75210, 'đŦ': 75830, 'đ': 76077, 'đ': 77151, 'đ': 98079, 'đ': 100212, 'đą': 100653, 'đ¤': 102889, 'đ': 105564, 'đ': 106974, 'đĒ': 107703, 'đĄ': 111627, 'đ': 115936, 'đĢ': 116199, 'đ': 120496, 'đ': 121236, 'đ': 122249, 'đ': 122721, 'đ': 127339, 'đ': 132498, 'đ': 136738, 'âē': 138523, 'đ': 139500, 'đĨē': 141039, 'đļ': 144650, 'đ': 146993, 'đ': 148373, 'đ': 153020, 'đ': 153827, 'đ': 155966, 'đĸ': 161518, 'đ': 166034, 'đ´': 170803, 'đ¤Ŗ': 175802, 'â¨': 180694, 'đ': 189629, 'đ': 194271, 'đ': 195752, 'âēī¸': 197675, 'đ': 213108, 'đ': 213792, 'đ': 215726, 'đ': 235897, 'đ': 251745, 'đ¤': 286776, 'đ': 293981, 'đŗ': 294057, 'đ': 304936, 'đ': 309821, 'đ': 310541, 'đ¯': 317351, 'đĨ': 332628, 'đ': 332689, 'đ': 343735, 'đ': 353626, 'â¤': 360275, 'âĨ': 386672, 'đ': 511117, 'đ': 612212, 'đŠ': 619148, 'â¤ī¸': 727927, 'đ': 1015346, 'đ­': 1302483, 'đ': 4128538}
emo_list = list(emo_100.keys())
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

with open('./sem19-task6-offen.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
keywords = []
for line in lines:
    line = ' '.join(line.strip().split(' ')[:-1])
    keywords.append(line)
# query = ' OR '.join(keywords[:50])

query_txt_no = '-is:retweet -RT -has:links -has:media -has:videos -has:images lang:en '
# badquery=' include:antisocial include:offensive_user include:antisocial_offensive_user'
# query_txt_cont = ' OR '.join(keywords[:50])#'(#irony OR #sarcasm OR #not)'
query_emo = ' ('+' OR '.join(emo_list[50:]) + ')'

all_tweet = []
for query_txt_cont in keywords:
    if 'filter' in query_txt_cont:
        continue
    tweets = Paginator(client.search_recent_tweets, query=query_txt_no+query_txt_cont+query_emo,
                                  #tweet_fields=['context_annotations', 'created_at'],
                                  max_results=100).flatten(1000)
    for tweet in tweets:
        all_tweet.append(tweet)
import json
with open('retrieve_sem19-task6-offen.json', 'a', encoding='utf-8') as f:
    for tweet in all_tweet:
        json.dump(tweet, f)
        f.write('\n')
