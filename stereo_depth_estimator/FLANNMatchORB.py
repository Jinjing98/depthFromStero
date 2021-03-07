import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt





def get_matched_idx_pos_disparity(img1_path,img2_path):
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)  # trainImage
    # Initiate SIFT detector
    # https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/
    # sift = cv.xfeatures2d.SIFT_create()
    # # sift = cv.SIFT_create()
    # # find the keypoints and descriptors with SIFT
    #
    # kp1, des1 = sift.detectAndCompute(img1,None)
    # kp2, des2 = sift.detectAndCompute(img2,None)

    orb = cv.ORB_create(nlevels=8)
    # nfeatures=None, scaleFactor=None, nlevels=None, nlevels The number of pyramid levels. The smallest level will have linear size equal to
    # .   input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
    # find the keypoints with ORB
    kp1 = orb.detect(img1, None)
    # compute the descriptors with ORB
    kp1, des1 = orb.compute(img1, kp1)
    kp2 = orb.detect(img2, None)
    # compute the descriptors with ORB
    kp2, des2 = orb.compute(img2, kp2)

    # get the LUT of pos For 2 imgs
    LUT_queryImg1 = []
    LUT_trainImg2 = []
    for i, n in enumerate(kp1):
        LUT_queryImg1.append((i, n.pt))
        # print('query idx:',str(i),' pos:', list(n.pt))   #0-2165
    LUT_queryImg1 = dict(LUT_queryImg1)
    print('\nQUERY LUT', LUT_queryImg1)

    for i, n in enumerate(kp2):
        LUT_trainImg2.append((i, n.pt))
        # print('train idx:',str(i),' pos:', list(n.pt))  #0-2683
    LUT_trainImg2 = dict(LUT_trainImg2)
    print('\nTRAIN LUT', LUT_trainImg2)





    # get the LUT of des For 2 imgs
    LUT_queryImg1_des = []
    LUT_trainImg2_des = []
    for i in range(np.shape(des1)[0]):
        LUT_queryImg1_des.append((i,list(des1[i])))
    LUT_queryImg1_des = dict(LUT_queryImg1_des)
    print('\nQUERY LUT of descriptor', LUT_queryImg1_des)

    for i in range(np.shape(des2)[0]):
        LUT_trainImg2_des.append((i,list(des2[i])))
    LUT_trainImg2_des = dict(LUT_trainImg2_des)
    print('\ntrain LUT of descriptor', LUT_trainImg2_des)












    # matching procudure
    # FLANN parameters    the methods used for matching.  another choice, BF
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)   # or pass empty dictionary
    # flann = cv.FlannBasedMatcher(index_params,search_params)
    # flann = cv.FlannBasedMatcher(index_params)

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    # print('\nthe num of unfimaltered matched kp pairs :' + str(len(matches)))

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    num_MP = 0
    matched_idx = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.6 * n.distance:  # top two distances m n, a smaller parameter means a higher precision n less matching
            matchesMask[i] = [1, 0]  # this mask encode connect the best m or the second best n
            num_MP += 1
            matched_idx.append([m.queryIdx, m.trainIdx])
    # print('the index for matched kp (query,train):',matched_idx)  #same   query is ordered.

    matched_idxNpos_obj = open('matched_idxNpos.txt', 'w')
    # matched_idxNpos_obj.write('\n')
    matched_idxNpos_obj.write(img1_path[-9:]+'\n')
    matched_idxNpos_obj.write('query kp idx/ query kp pos / train kp idx / train kp pos / dispartity / query des / train des\n ')
    matched_idxNpos = []
    for i in matched_idx:
        query_idx, train_idx = i[0], i[1]
        matched_idxNpos.append([query_idx, LUT_queryImg1[query_idx], train_idx, LUT_trainImg2[train_idx]])
        matched_idxNpos_obj.write(str([query_idx, LUT_queryImg1[query_idx], train_idx, LUT_trainImg2[train_idx],
                                       LUT_queryImg1[query_idx][0] - LUT_trainImg2[train_idx][0],LUT_queryImg1_des[query_idx],LUT_trainImg2_des[train_idx]]))
        matched_idxNpos_obj.write('\n')
    matched_idxNpos_obj.write('\n')
    matched_idxNpos_obj.close()

    print('\nthe index N pos for matched kp (query,train):', matched_idxNpos)  # same   query is ordered.

    draw_params = dict(matchColor=(255, 218, 185),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    print('\nthe num of orb kp in query_img img1 :' + str(len(kp1)))
    print('the num of orb kp in train_img img2 :' + str(len(kp2)))
    print('the num of filtered matched kp pairs :' + str(num_MP))

    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    # img2 = cv.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)

    cv.namedWindow('', 0)  # so that the img will be clipped
    cv.imshow('', img3)

    cv.waitKey(0)
path1 ='data/1/left/00001.png'
path2 ='data/1/right/00001.png'
get_matched_idx_pos_disparity(path1,path2)
