# get the csv file with the predicted counts and actual and get the difference and graph it?
# step one:
# put the actual img tg
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import matplotlib.patches as patches

df_names = pd.read_csv("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/FullCapsSet/sj2_training_sums.csv")

IMG_NAMES = df_names['RootImage'].unique()

df = pd.read_csv("/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/FullCapsSet/sj2_training_indivs.csv")

names = df['ImageName']

for IMG_NAME in IMG_NAMES:
    images_i_want = []
    for i in names:
        name = i.split('count')[1].split('pt')[0][:-1]
        if IMG_NAME == name:
            # print(i)
            images_i_want.append(i)

    images_i_want = sorted(images_i_want)

    DS_PTH = "/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/SilkyJohnson2"

    df_new = pd.read_csv('/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/FullCapsSet/sj2_caps.csv')
    fig, ax = plt.subplots(figsize=(6, 6))
    for index, row in df_new.iterrows():
        if row['Filename'] in images_i_want:
            #print(row['Filename'])
            cnt = int(row['Filename'].split('eggs')[1].split('count')[0])
            full_path = f'{DS_PTH}/{cnt}/{row['Filename']}'
            #print(full_path)

            # Load the image
            img = mpimg.imread(full_path)

            # Create a figure and axes
            x_coord = row['x']
            y_coord = 10-row['y']
            #print(x_coord, y_coord)

            # Create an OffsetImage object with the image data
            imagebox = OffsetImage(img, zoom=0.45)  # Adjust zoom as needed

            # Create an AnnotationBbox to place the image at the specified coordinates
            ab = AnnotationBbox(imagebox, (x_coord, y_coord), frameon=False)

            ax.add_artist(ab)
            #print(ax)

            ax.set_xlim(0, 10) 
            ax.set_ylim(0, 10)
            # plt.xticks(np.arange(0, 10, step=1))
            # plt.yticks(np.arange(0, 10, step=1))
            plt.grid(False)
            plt.axis('off')
            plt.plot()
            plt.margins(x=0)
        
        plt.show()

    # Show the plot
    value = int(df_names.loc[df_names['RootImage'] == IMG_NAME, 'Sum'].values)
    try:
        os.mkdir(f'/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/FullCaps/{value}')
    except FileExistsError:
        pass
    plt.savefig(f"/home/drosophila-lab/Documents/Fecundity/CNN-Classifier/TrainingSets/FullCaps/{value}/{IMG_NAME}_reconstructed", 
                bbox_inches='tight')


# step two:
# apply the colors to it and reconstruct