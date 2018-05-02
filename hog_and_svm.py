import cv2
import numpy as np
import math
from os import listdir
from os.path import isfile, join
from random import shuffle


letters_10 = ["a", "i", "u", "e", "o", "ha", "hi", "fu", "he", "ho"]
letters_half = ["a","i","u","e","o","ka","ki","ku","ke","ko","sa","shi",
                "su","se","so","ha","hi","fu","he","ho","ma","mi","mu",
                "me","mo","n","na","ni","nu","ne","no","ra","ri","ru","re",
                "ro","ta","chi","tsu","to","te","wa","wo","ya","yo","yu"]

letters_full = ["a","i","u","e","o","ka","ki","ku","ke","ko","sa","shi","su","se","so",
                "ha","hi","fu","he","ho","ma","mi","mu","me","mo","n",
                "na","ni","nu","ne","no","ra","ri","ru","re","ro",
                "ta","chi","tsu","te","to","wa","wo","ya","yo","yu",
                "da", "ji", "du", "de", "do","za","ji(shi)","zu","ze","zo",
                "ba","bi","bu","be","bo","pa","pi","pu","pe","po", "ga","gi","gu","ge","go"]

#print len(letters_full)

def readImageList(letters, first_number,last_number):
    img_list = []
    for letter in letters:
        for i in range(first_number, last_number):
            file_name = "test-data/" + letter + "/" + str(i) + "_50x50.png"
            img = cv2.imread(file_name, 0)
            img_list.append(img)
    each_num = last_number - first_number
    lettersN = len(letters)
    return img_list, lettersN, each_num

def readImageListTrain(lettersList):
    img_list = []
    folder = "train_case/"
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    train_files = []
    for letter in lettersList:
        count = 0
        nameFiles = [k for k in files if k.startswith(letter + "_")]
        shuffle(nameFiles)
        for fileName in nameFiles:
            image = cv2.imread(folder + fileName, 0)
            #print folder+fileName
            img_list.append(image)
            train_files.append(fileName)
            count += 1
            #save_file = "train_case/"
            #cv2.imwrite(save_file + fileName, image)
            #if count >= each_num:
            #    break
    each_num = count
    lettersN = len(lettersList)
    return img_list, lettersN, each_num, train_files

def readImageListTest(lettersList, each_num, train_files = []):
    img_list = []
    folder = "test-data/singles_50x50/"
    files = [f for f in listdir(folder) if isfile(join(folder, f)) and f not in train_files]
    for letter in lettersList:
        count = 0
        nameFiles = [k for k in files if k.startswith(letter + "_")]
        shuffle(nameFiles)
        for fileName in nameFiles:
            image = cv2.imread(folder + fileName, 0)
            #print fileName
            img_list.append(image)
            count += 1
            if count >= each_num:
                break
    lettersN = len(lettersList)
    return img_list, lettersN, each_num


def imgProcesSingle(img):
    img = cv2.resize(img, (50,50))
    return img

def imgProcess(image_list):
    output_img_list = []
    for img in image_list:
        img = imgProcesSingle(img)
        output_img_list.append(img)
    return output_img_list

def labelGen(lettersN, each_num):
    return np.repeat(range(lettersN), each_num)

def to1DArray(array):
    output_array = []
    for data in array:
        tmp = data[0]
        output_array.append(tmp)
    output_array = np.array(output_array)
    return output_array

def hog_compute(img_list):
    winSize = (50, 50)
    blockSize = (20, 20)
    blockStride = (5,5)
    cellSize = (5, 5)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture,
                            winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    hog_list = []
    for img in img_list:
        hist = hog.compute(img)
        hist = to1DArray(hist)
        #print "hist len: " + str(len(hist))
        hog_list.append(hist)
    hog_list = np.array(hog_list)
    #print "len hist: " + str(len(hog_list[0]))
    return hog_list

def hog_compute_opt(img_list, cellSize, blockSize, blockStride = (5,5)):
    winSize = (50, 50)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture,
                            winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    hog_list = []
    for img in img_list:
        hist = hog.compute(img)
        hist = to1DArray(hist)
        #print "hist len: " + str(len(hist))
        hog_list.append(hist)
    hog_list = np.array(hog_list)
    #print "len hist: " + str(len(hog_list[0]))
    return hog_list

def svmSetUp(svm_c):
    svm = cv2.ml.SVM_create()
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(svm_c)
    #svm.setGamma(5.1)
    return svm

def svmTrain(svm, hog_list, labels):
    svm.train(hog_list, cv2.ml.ROW_SAMPLE, labels)
    return True

def svmPredict1Image(svm, file_name, lettersList):
    img = cv2.imread(file_name, 0)
    img = imgProcesSingle(img)
    imgListTmp = []
    imgListTmp.append(img)
    test = hog_compute(imgListTmp)
    #print "test len: " + str(len(test[0]))
    result = svm.predict(test)
    #print result[1][0]
    index = int(result[1][0])
    return lettersList[index]

def accuracyTest(svm, lettersList, eachnum, train_files = []):
    img_list, lettersN, eachN = readImageListTest(lettersList, eachnum, train_files)
    img_list = imgProcess(img_list)
    test_list = hog_compute(img_list)
    test_labels = labelGen(lettersN, eachN)
    #print test_labels
    result = svm.predict(test_list)
    #print "Numbers of test: " + str(result[1].size)
    mask = result[1] == test_labels
    correct = 0
    for i in range(len(test_labels)):
        if test_labels[i] == result[1][i]:
            correct += 1
    #print "Number of correct: " + str(correct)
    accuracy = float(correct * 100) / result[1].size
    #print "Accuracy = " + str(accuracy)
    return accuracy

def accuracyOnLetter(svm, lettersList, letter, each_num, train_files = []):
    img_list = []
    folder = "test-data/singles_50x50/"
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    count = 0
    nameFiles = [k for k in files if k.startswith(letter + "_")]
    shuffle(nameFiles)
    for fileName in nameFiles:
        image = cv2.imread(folder + fileName, 0)
        #print folder + fileName
        img_list.append(image)
        count += 1
        if count >= each_num:
            break
    i = lettersList.index(letter)
    #print i
    lettersN = len(lettersList)
    img_list = imgProcess(img_list)
    test_list = hog_compute(img_list)
    test_labels = np.repeat(i, each_num)
    result = svm.predict(test_list)
    #print result[1].size
    mask = result[1] == test_labels
    correct = 0
    for i in range(len(test_labels)):
        if test_labels[i] == result[1][i]:
            correct += 1
    #print correct
    accuracy = float(correct * 100) / result[1].size
    #print "Accuracy on letter " + letter +" with " + str(each_num) + " letters = " + str(accuracy)
    return accuracy
