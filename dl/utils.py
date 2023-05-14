import csv
import json
import os
import sys


def csv_to_json(csv_file_path, json_file_path):
    # create a dictionary
    data_dict = {}

    # Step 2
    # open a csv file handler
    with open(csv_file_path, encoding='utf-8') as csv_file_handler:
        csv_reader = csv.DictReader(csv_file_handler)

        # convert each row into a dictionary
        # and add the converted data to the data_variable

        for row in csv_reader:
            # assuming a column named 'No'
            # to be the primary key
            key = row['']
            data_dict[key] = {k: v for k, v in row.items() if k != ""}

    # open a json file handler and use json.dumps
    # method to dump the data
    # Step 3
    with open(json_file_path, 'w', encoding='utf-8') as json_file_handler:
        # Step 4
        json_file_handler.write(json.dumps(data_dict, indent=4))


if __name__ == "__main__":
    csv_file_path = os.getcwd() + '/preprocessed.csv'
    json_file_path = os.getcwd() + '/preprocessed.json'

    max_int = sys.maxsize

    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

    csv.field_size_limit(max_int)
    csv_to_json(csv_file_path, json_file_path)
