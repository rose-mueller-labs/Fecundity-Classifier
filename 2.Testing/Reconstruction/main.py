from generate_x_y import main
from visual import generate_vis
import pandas as pd

model = "4-30_5-1_CC_A"
images_i_want = ["3 3 D4 16.jpg"]

if __name__ == '__main__':
    model_name = f"{model}_v0.0"
    csv_name = "x_y_parts.csv"
    tiles_csv = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/2.Testing/model_testing_complete_CD_results/{model_name}_tile_counts_CD_Complete.csv"
    main(csv_name, tiles_csv)
    ROOT_IMG = "/home/drosophila-lab/Documents/Fecundity/Fecundity-Classifier/DATA/ALL_CD_CAPS-sliced"
    df_new = pd.read_csv(csv_name)
    generate_vis(df_new, images_i_want, ROOT_IMG)