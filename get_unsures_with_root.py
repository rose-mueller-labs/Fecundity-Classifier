# SRC = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/UsableData"
# DEST = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/DATA/Unsures-5-1-5-4-4-29-Wint"
# import os
# import shutil

# # txt file that lists the whole images' names that we want
#     # will retrieve them manually?
# # all the unsures go into one dir

# root_img_names = []

# for day_dir in os.listdir(SRC):
#     day_dir_p = f'{SRC}/{day_dir}'
#     day = None
#     if 'cap' in day_dir:
#         day = day_dir.split('-cap')[0]
#     else:
#         day = "Winter 2017 2 21 C pops"
#     for img in os.listdir(day_dir_p):
#         if 'unsure' not in img:
#             continue
#         root_image_a = img.split('count')[1]
#         root_image_b = root_image_a.split('pt')
#         root_image = root_image_b[0]
#         root_image = root_image.strip()
#         root_img_names.append(f'{day}: {root_image}')

#         shutil.move(f'{day_dir_p}/{img}', f'{DEST}/{img}')

# with open('root_images_needed.txt', 'w') as f:
#     for i in root_img_names:
#         print(f"{i}\n", file=f)

DEST = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/DATA/Unsures-5-1-5-4-4-29-Wint"


import os
import shutil 

# -----------
# Note for Alden: replace these with the actual folders with the caps in them and run the script here

P_54 = "/home/drosophila-lab/Downloads/5-4-cap-20250528T193636Z-1-001/5-4-cap"

P_51 = "/home/drosophila-lab/Downloads/5-1-cap-20250528T193632Z-1-001/5-1-cap"

P_Wint = None

P_430 = None
# ------------
SRC = "/home/drosophila-lab/Documents/All Lithium Caps"

with open('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/DATA/Unsures-5-1-5-4-4-29-Wint-sliced/root_images_needed.txt', 'r') as f:
    for line in f:
        if line == "\n":
            continue
        try:
            root_img = str(line).split(':')[1].strip()
        except IndexError:
            print(line)
            raise Exception
        for img in os.listdir(SRC):
            if f"{root_img}.png" in img:
                shutil.copy(f'{SRC}/{img}', f'{DEST}/{img}')