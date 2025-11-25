# from generate_x_y import main
from big_generate_x_y import main
# from visual import generate_vis
from big_visual import generate_vis
import pandas as pd

for model in ["Alex_FecundityModelMoDataV1"]: # ["Alex_5-1_5-2S", "Alex_4-30_5-1_CC_A", "Alex_5-1_5-2S_CC_A"]:
    # model = "Alex_5-1_5-2S" #"Alex_4-30_5-1_CC_A"
    # ["3 3 D4 16.jpg", "2 28 C1 32.jpg"]
    csv_name = "big_x_y_parts.csv"
    model_name = f"{model}_v0.0"
    tiles_csv = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/4.FullPipeline/SecondBigCap_Alex_FecundityModelMoDataV1_tile_counts_cage_testing.csv"
    # tiles_csv = f"/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_complete_CD_results/{model_name}_tile_counts_CD_Complete.csv"
    # main(csv_name, tiles_csv)
    
    # ROOT_IMG = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/ALL_CD_CAPS-sliced"
    ROOT_IMG = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/SecondBigCap-half-sliced"
    df_new = pd.read_csv(csv_name)
    # images_i_want = ["3 3 D5 44.jpg", "3 3 D4 16.jpg", "2 28 C1 32.jpg"]
    images_i_want = ["IMG_0010.JPG", "IMG_0006.JPG", "IMG_0054.JPG"]
    print(images_i_want)
    generate_vis(df_new, images_i_want, ROOT_IMG, model_name)