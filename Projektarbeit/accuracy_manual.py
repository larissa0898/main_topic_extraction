import json

f = open('fromhand.json', encoding='utf-8')
fromhand = json.load(f)


title = list(fromhand.keys())
keywords = list(fromhand.values())




realcount = 0

for i in range(len(title)):
    count = 0
    for key in keywords[i]:
        if key in title[i]:
            count += 1
    if count >= 1:
        realcount += 1

print(realcount/247*100)