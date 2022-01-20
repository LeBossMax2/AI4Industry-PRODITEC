import cv2 as cv
import numpy as np
import os

mypath='data/DATASET1_CROPPED_SORTED/'
mypathsave = "data/IA_test/"
if not os.path.exists(mypathsave):
    os.mkdir(mypathsave)
folders = os.listdir(mypath)
for folder in folders:

    onlyfiles2 = os.listdir(mypath+str(folder))
    link = str(folder)

    if not os.path.exists(mypathsave+link):
        os.mkdir(mypathsave+link)

    for n in range(0, len(onlyfiles2)):
        
        print(mypath+link+"/"+onlyfiles2[n])
        src = cv.imread( mypath+link+"/"+onlyfiles2[n] )
        if src is None:
            continue

        height, width = src.shape[:2]

        #src = src[90:height-90,170:width-170]

        #src = cv.resize(src,(223,223))

        cv.imwrite(mypathsave+link+"/"+str(n)+"-1.jpg",src)

        #height, width = src.shape[:2]

        center = (width/2, height/2)

        rotate_matrix = cv.getRotationMatrix2D(center=center, angle=90, scale=1)

        rotated_image = cv.warpAffine(src=src, M=rotate_matrix, dsize=(width, height))

        cv.imwrite(mypathsave+link+"/"+str(n)+"-2.jpg",rotated_image)

        rotated_image = cv.warpAffine(src=rotated_image, M=rotate_matrix, dsize=(width, height))

        cv.imwrite(mypathsave+link+"/"+str(n)+"-3.jpg",rotated_image)

        rotated_image = cv.warpAffine(src=rotated_image, M=rotate_matrix, dsize=(width, height))

        cv.imwrite(mypathsave+link+"/"+str(n)+"-4.jpg",rotated_image)