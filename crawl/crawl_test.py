import tweepy
consumer_key="b0qF9CcEslZWdfTY6Gyfr6k7i"
consumer_secret="x8tKpNAUbeiCaTCFyIBmDvpXHsCD91ZOx0MHq4rU7ZN92yZI4k"
access_token="1517497475749539841-9LVF5opQbfkuUnreMz5v3PsyxZCID1"
access_token_secret="Ok2vxvbvJWS579a3fAgjkytr5iVoZMWwVUGMLr5FkhJcQ"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

try:
    api.verify_credentials()
    print('Successful Authentication')
except:
    print('Failed authentication')