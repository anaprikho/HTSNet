from skimage import img_as_float, img_as_ubyte
from skimage.filters import threshold_sauvola
from skimage import img_as_float
from skimage.color import gray2rgb
import skimage.io as io
from skimage import data
import glob
import cv2
import os
import numpy as np


SAVE_DIR = 'C:/Users/prikha/Downloads/BA/Datasets/HTSNet_Syn_method/bin_m/'

hand = glob.glob('C:/Users/prikha/Downloads/BA/Datasets/HTSNet_Syn_method/VoTT_color/handwritten_crops/*.jpg')
docs = glob.glob('C:/Users/prikha/Downloads/BA/Datasets/HTSNet_Syn_method/VoTT_color/machineprinted_crops/*.jpg')

BOXWDITH = 256
STRIDE = BOXWDITH - 10



def getbinim(img):
    # if len(image.shape) >= 3:
    #     image = rgb2gray(image)
    # thresh_sauvola = threshold_sauvola(image)
    # binary_image = img_as_float(image > thresh_sauvola)
    # return image

    height, width, channels = img.shape

    # Create blank Binary Image
    img_binary = np.zeros((height, width, 1))

    # Create grayscale image
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print img_grayscale.shape
    (thresh, img_binary) = cv2.threshold(img_grayscale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_binary



    # t_sauvola = threshold_sauvola(image, window_size=15)
    # binary_image = image > t_sauvola
    # return binary_image

    #return t_sauvola

    #im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # th, im_gray_th_otsu = cv2.threshold(im_gray, 128, 192, cv2.THRESH_OTSU)
    # return im_gray_th_otsu

    # ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # return bw_img



    # ------------This one is working: ---------------------
    # img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # img_grey = cv2.IMREAD_GRAYSCALE(image)
    # thresh = 100
    # img_binary = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)[1]
    # return img_binary


if __name__ == "__main__":
    output_folder = SAVE_DIR

    i = 0
    for img in docs:
        img = cv2.imread(img)
        img = (getbinim(img))

        os.chdir(SAVE_DIR)
        cv2.imwrite(output_folder + str(i) + 'out.jpg', img)
        i = i + 1