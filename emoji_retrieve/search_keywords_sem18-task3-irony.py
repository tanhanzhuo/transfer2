import tweepy
from pagination import Paginator
import os
emo_100={'😠': 39968, '🗣': 41421, '🌚': 42538, '®': 43092, '😶': 43145, '🙏🏾': 43797, '👑': 44462, '😖': 46428, '😥': 46615, '✌️': 46644, '☹️': 46909, '💚': 48557, '©': 49189, '👊': 49252, '🥰': 49539, '💓': 55596, '🙊': 57845, '😻': 59191, '📷': 60233, '💪': 60579, '✋': 60885, '😣': 60888, '🇺🇸': 61952, '✌': 63647, '‼️': 63962, '💘': 64140, '😇': 64764, '😛': 64790, '😃': 64805, '😷': 66930, '😓': 67046, '💛': 67355, '🤗': 71526, '🙂': 72717, '💞': 72994, '😝': 73519, '💋': 75210, '😬': 75830, '😀': 76077, '😆': 77151, '💁': 98079, '🎉': 100212, '😱': 100653, '😤': 102889, '😄': 105564, '💔': 106974, '😪': 107703, '😡': 111627, '💗': 115936, '😫': 116199, '😞': 120496, '🙈': 121236, '😈': 122249, '😜': 122721, '💖': 127339, '💙': 132498, '😋': 136738, '☺': 138523, '😐': 139500, '🥺': 141039, '🎶': 144650, '💜': 146993, '👏': 148373, '😑': 153020, '🙃': 153827, '😕': 155966, '😢': 161518, '👍': 166034, '😴': 170803, '🤣': 175802, '✨': 180694, '🙌': 189629, '🙏': 194271, '😎': 195752, '☺️': 197675, '😌': 213108, '😉': 213792, '😅': 215726, '💀': 235897, '👀': 251745, '🤔': 286776, '🙄': 293981, '😳': 294057, '👌': 304936, '😁': 309821, '😏': 310541, '💯': 317351, '🔥': 332628, '💕': 332689, '😘': 343735, '😔': 353626, '❤': 360275, '♥': 386672, '😒': 511117, '😊': 612212, '😩': 619148, '❤️': 727927, '😍': 1015346, '😭': 1302483, '😂': 4128538}
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

query_emo = ' ('+' OR '.join(emo_list[36:]) + ')'
query_txt = '-is:retweet -RT -has:links -has:media -has:videos -has:images lang:en (#irony OR #sarcasm OR #not)'

all_tweet = []
tweets = Paginator(client.search_recent_tweets, query=query_txt+query_emo,
                              #tweet_fields=['context_annotations', 'created_at'],
                              max_results=10).flatten()
for tweet in tweets:
    all_tweet.append(tweet)
import json
with open('retrieve_sem18-task3-irony.json', 'w', encoding='utf-8') as f:
    for tweet in all_tweet:
        json.dump(tweet, f)
        f.write('\n')