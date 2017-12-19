import matplotlib
import numpy as np
import cv2
from matplotlib import pyplot as plt
#import PyQt4
matplotlib.get_backend()
#matplotlib.use('qt4agg')
matplotlib.matplotlib_fname()
imgL = cv2.imread('Frames/drive_cam0_0067.bmp', 0)
imgR = cv2.imread('Frames/drive_cam1_0067.bmp', 0)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()
plt.savefig('res.png')
