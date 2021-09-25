import glob
import cv2
import os
import numpy as np

SAVE_DIR = 'D:/Uni/BA/Datasets/IAM/USED/test_binar/'

IM_DIR = glob.glob('D:/Uni/BA/Datasets/HTSNet_data_synthesis/wgm/VoTT_color(old)/handwritten_crops/*')


def binim(img):
    """
    Takes an image file as input and returns a binarized image.
    :type img: input image file
    """
    # Create grayscale image
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (thresh, img_binary) = cv2.threshold(img_grayscale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_binary


if __name__ == "__main__":

    i = 0
    for img in IM_DIR:
        img = cv2.imread(img)
        img = (binim(img))

        os.chdir(SAVE_DIR)
        cv2.imwrite(SAVE_DIR + str(i) + 'out.jpg', img)
        i = i + 1