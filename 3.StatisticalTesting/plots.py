import os
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr

# Create a histogram that shows the distribution of eggs in the training sets
def make_hist():
    BASE_DIR = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/1.DataProcessing/DATASETS"
    accepted = ['4-30_5-1_5-2O_CC_A', 
                '4-30_5-1_5-2O',
                '4-30_5-1_5-2S',
                '4-30_5-1_CC_A',
                '4-30_5-1',
                '4-30_5-2O', '4-30_5-2S',
                '4-30', '5-1_5-2O',
                '5-1_5-2S_CC_A',
                '5-1_5-2S', '5-1', '5-2O', 
                '5-2S', 'CC_A_4-30', 'CC_A', 'CC_J',
                '4-30_5-1_5-2S_CC_A_CC_J',
                '4-30_5-1_CC_A_CC_J', 
                'GS_4-30_5-1_5-2O_CC_A']

    training_set_egs = dict() # key = dataset_name,value = dict(key = eggcnt, value = # of imgs)

    for data_dir_name in os.listdir(BASE_DIR):
        if data_dir_name not in accepted:
            continue
        training_set_egs[data_dir_name] = dict()
        for egg_cnt in os.listdir(f"{BASE_DIR}/{data_dir_name}"):
            # print(egg_cnt)
            training_set_egs[data_dir_name][int(egg_cnt)] = len(os.listdir(f"{BASE_DIR}/{data_dir_name}/{egg_cnt}"))
    # print(training_set_egs)
    to_plot = set(training_set_egs.keys())

    fig, axes = plt.subplots(5, 4, figsize=[35, 28])
    for row in range(5):
        for col in range(4):
            ds = to_plot.pop()
            # x = list(training_set_egs[ds].keys())
            # y = list(training_set_egs[ds].values())
            # print(training_set_egs[ds])
            axes[row, col].bar(training_set_egs[ds].keys(), training_set_egs[ds].values(), width=1, ec='k')
            
            title = ds.replace('_', ',')
            # print(title)
            if 'CC,A' in title:
                title = title.replace('CC,A', 'CC_A')
            if 'CC,J' in title:
                title = title.replace('CC,J', 'CC_J')
            if 'GS,' in title:
                title  = title.replace('GS,', 'Grayscale: ')
            axes[row, col].set_title(f'{title} Egg Distribution', fontsize=24)
            
            axes[row, col].set_xlabel('Tile Egg Count', fontsize=18)
            axes[row, col].set_ylabel('Frequency', fontsize=18)
            axes[row, col].tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.savefig('hist.png')

# Calculate or describe the Cohen’s Kappa or Pearson correlation between Julie and Angela's Counts

def get_cohen_and_pearson():
    ANGELA = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/5-4-cap-sliced-Angela"
    JULIE = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/5-4-cap-sliced-Julie"

    # DF with column1 as filename and columns 2 and 3 as ANGELA and JULIE's count

    angela_counts = dict()
    julie_counts = dict()

    for filename in os.listdir(ANGELA):
        if 'eggs' not in filename or 'unsure' in filename:
            continue
        # print()
        cnt = filename.split('eggs')[1].split('count')[0]
        name = filename.split('count')[1]
        angela_counts[name] = cnt
    for filename in os.listdir(JULIE):
        if 'eggs' not in filename or 'unsure' in filename:
            continue
        cnt = filename.split('eggs')[1].split('count')[0]
        name = filename.split('count')[1]
        julie_counts[name] = cnt

    aligned_dict = {k: (angela_counts[k], julie_counts[k]) for k in angela_counts if k in julie_counts}

    y1 = [v[0] for v in list(aligned_dict.values())]
    y2 = [v[1] for v in list(aligned_dict.values())]

    print("Cohen's Kappa: ", cohen_kappa_score(y1, y2))
    res = pearsonr(y1, y2)
    res = res.confidence_interval(confidence_level=0.9)

    print(f"Pearson Correlation: {res}")