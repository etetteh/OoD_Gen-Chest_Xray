import os
import csv

def write_results(filename: str, field_names: list, results: dict):
    read_file = open(filename, "r")
    results_file = csv.DictReader(read_file)
    update = []
    new = []
    row = {}
    for r in results_file:
        if r['Model'] == results['Model']:
            for key, value in results.items():
                row[key] = value
            update = row
        else:
            for key, value in results.items():
                row[key] = value
            new = row
    
    read_file.close()

    if update:
        print("Results exists. Updating results in file...")
        print(update)
        update_file = open(filename, "w", newline='')
        data = csv.DictWriter(update_file, delimiter=',', fieldnames=field_names)
        data.writeheader()
        data.writerows([update])
    else:
        print("Results does not exist. Writing results to file...")
        print(new)
        update_file = open(filename, "a+", newline='')
        data = csv.DictWriter(update_file, delimiter=',', fieldnames=field_names)
        data.writerows([new])
    
    update_file.close()

