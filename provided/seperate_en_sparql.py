import json
import argparse


def seperate(data_dir, subset, output_dir):

    data_dir = data_dir.rstrip("/")
    output_dir = output_dir.rstrip("/")

    en_file = open(f"{output_dir}/{subset}.en", "w+", encoding="utf-8")
    sparql_file = open(f"{output_dir}/{subset}.sparql", "w+", encoding="utf-8")
    data = json.load(open(f"{data_dir}/{subset}.json", "r", encoding="utf-8"))
    for element in data:
        if element["question"]:
            en_file.write(element["question"] + "\n")
            sparql_file.write(element["sparql_wikidata"] + "\n")

    en_file.close()
    sparql_file.close()
