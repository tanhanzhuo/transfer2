def readhash(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        hashtags = []
        for line in lines:
            hashtags.append(line.strip().split('\t'))

hash_select = readhash('hash_select.txt')
