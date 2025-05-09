import numpy as np
import cv2 as cv
import math
import os
import sys

H = 750
W = 750

# Given a square image, turns it into 100 splits of 75x75.
def split_image(outpath, imgpath, imgid):
    print(f"SPLITTING {imgid}")
    img = cv.imread(imgpath, 1)
    img750 = cv.resize(img, (H, W), interpolation=cv.INTER_AREA)
    index = 1
    imgname = imgid[0:imgid.find(".")]
    # Since not all image resolutions are evenly divisible by 75, center w/ indent:
    y_splits = 10
    x_splits = 10
    for c in range(x_splits): # the number of columns = x
        for r in range(y_splits): # the number of rows = y
            y1 = r*75 
            y2 = (r+1)*75 
            x1 = c*75 
            x2 = (c+1)*75 
            crop_img = img750[y1:y2, x1:x2] 
            # cv.imwrite(r'{}/{} pt{}.jpg'.format(outpath,imgname,index),crop_img) 
            cv.imwrite(os.path.join(outpath, f'{imgname} pt{index}.jpg'), crop_img)
            # outputs jpg
            index += 1

def main(argv): 
    path = ""
    for i in range(1, len(sys.argv)):
        path += sys.argv[i] + " "
    path = path[0:len(path)-1]
    for x in os.listdir(path): 
        # imgpath = f'{path}/{x}'
        imgpath = os.path.join(path, x)
        print(imgpath)
        if  (x.endswith(".jpg") or x.endswith(".png")) and not x.startswith("."):
            outpath = r'{}-sliced'.format(path)
            if not os.path.exists(outpath):
                os.mkdir(outpath)
            split_image(outpath, imgpath, x)
    print("Finished!")

main(0) # to run, type: > python3 image_shredder_750.py [folder-path]

