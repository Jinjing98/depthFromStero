import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('data/1/left//00000.png',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('data/1/right/00000.png',cv.IMREAD_GRAYSCALE)         # trainImage
# Initiate SIFT detector
# https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/
# sift = cv.xfeatures2d.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)


orb = cv.ORB_create()
# find the keypoints with ORB
kp1 = orb.detect(img1,None)
# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2 = orb.detect(img2,None)
# compute the descriptors with ORB
kp2, des2 = orb.compute(img2, kp2)





# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)  #find the k best matches
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.namedWindow('', 0)#  so that the img will be clipped
cv.imshow('',img3)

cv.waitKey(0)