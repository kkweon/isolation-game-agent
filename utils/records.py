import csv

def save(filename, data):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['name', 'win_ratio'])
        for row in data:
            writer.writerow(row)