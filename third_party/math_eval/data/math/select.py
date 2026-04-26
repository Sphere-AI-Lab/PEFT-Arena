import json
import random
data_list = []
with open("train.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        data_list.append(data)

level_list = {}
for data in data_list:
    level = data["level"]
    if level not in level_list:
        level_list[level] = []
    level_list[level].append(data)

for level in level_list:
    print(level, len(level_list[level]))

select_data = []
for level in level_list:
    if len(level_list[level]) > 50:
        select_data.extend(random.sample(level_list[level], 50))

with open("select_data.jsonl", "w") as f:
    for data in select_data:
        f.write(json.dumps(data) + "\n")