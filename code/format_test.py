import json

with open("../dataset/word-test.v1.txt", "r", encoding="utf-8") as f:
    data = f.readlines()

analysis_type = None
dict_data = {}
for line in data:
    line = line.strip()

    if len(line) == 0:
        continue
    elif line.startswith("//"):
        continue
    elif line.startswith(": "):
        analysis_type = line.replace(": ", "")
        dict_data[analysis_type] = {}
    else:
        line = line.split()
        base = (line[0], line[1])
        if base not in dict_data[analysis_type]:
            dict_data[analysis_type][base] = []
        dict_data[analysis_type][base].append((line[2], line[3]))

json_data = {}
for analysis_type in dict_data:
    json_data[analysis_type] = []
    for base, value in dict_data[analysis_type].items():
        json_data[analysis_type].append({"key": base, "value": value})

with open("formatted.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)


# track the evolution of this analysis in tensorboard
# save the tb links about it to not save this in mind
# go to another step
# FINISH THIS WEEKEND! start new repo this weekend

# primeiro fazer como vai ser a conta matricial, depois formatar esse arquivo do modo mais apropriado, pois acho que da para fazer toda essa conta de uma vez só
