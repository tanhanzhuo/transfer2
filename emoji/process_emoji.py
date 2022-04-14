import emoji
data_emoji_pred = []
with open('data_emoji.txt', 'r') as f:
    for line in tqdm(f):
        emoji_one = emoji.distinct_emoji_list(line)[0]
        line_clean = emoji.replace_emoji(line)
        data_emoji_pred.append(emoji_one + '\t' + line_clean)

with open('data_emoji_pred.txt', 'r') as f:
