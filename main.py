corpus = []
lines = open('corpus.txt').readlines()
for i in range(20000):
    line = lines[i]
    tags = line.strip().split(' ')
    word = tags[0]
    pri = float(tags[1])
    corpus.append(word)

lines = open('phrases.txt').readlines()
for line in lines:
    line = line.strip()
    tags = line.split()
    flag = True
    for word in tags:
        if word not in corpus:
            flag = False
    if flag:
        print(line)
