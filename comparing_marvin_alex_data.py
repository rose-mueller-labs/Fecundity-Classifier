import os
import csv
import pandas as pd

MARVIN = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/5-1-cap-800x800-sliced-Marvin"
ALEX = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/5-1-cap-800x800-sliced-Alexander"

def get_raw(file):
    raw_file_name = file.split('eggs')[1]
    raw_file_name_2 = raw_file_name.split('count')[1]
    count = raw_file_name.split('count')[0]
    return count, raw_file_name_2

all_files = os.listdir(ALEX)

def get_alex_file(file, all_files):
    for named in all_files:
        if file == get_raw(named)[1]:
            if 'unsure' in named:
                return None, None
            alexcount, rawfile = get_raw(named)
            return alexcount, named

count_same = 0
total = 0
count_diff = 0
with open('5-1_marvin_alex_comparison_correct.csv', "w", newline='') as cmp_file:
    writer = csv.writer(cmp_file)
    writer.writerow(['FileName', 'MarvinFileName', 'AlexFileName', 'AlexCount', 'MarvinCount', 'Mean', 'Difference'])
    for file in os.listdir(MARVIN):
        if 'eggs0' in file or 'unsure' in file:
            continue
        path_name = f"{MARVIN}/{file}"
        
        count, raw_file_name = get_raw(file)
        total += 1
        if f'{file}' in all_files:
            count_same+=1
            alex_count, alex_filename = get_alex_file(raw_file_name, all_files)
            count = int(count)
            writer.writerow([raw_file_name, file, alex_filename, count, count, ((int(count)+int(count))/2), count-count])
        else:
            count_diff += 1
            alex_count, alex_filename = get_alex_file(raw_file_name, all_files)
            if (alex_count == None):
                continue
            count = int(count)
            alex_count = int(alex_count)
            writer.writerow([raw_file_name, file, alex_filename, alex_count, count, ((count + alex_count)/2), count-alex_count])
            print(alex_filename)
            print(alex_count)
            print(count)
            print(file)
            print('-----')


print('same', count_same)
print('diff', count_diff)
print('total', total)