import cv2
import numpy as np
import pandas as pd


letters_10 = ["a", "i", "u", "e", "o", "ha", "hi", "fu", "he", "ho"]
letters_half = ["a","i","u","e","o","ka","ki","ku","ke","ko","sa","shi",
                "su","se","so","ha","hi","fu","he","ho","ma","mi","mu",
                "me","mo","n","na","ni","nu","ne","no","ra","ri","ru","re",
                "ro","ta","chi","tsu","to","te","wa","wo","ya","yo","yu"]

letters_full = ["a","i","u","e","o","ka","ki","ku","ke","ko","sa","shi","su","se","so",
                "ha","hi","fu","he","ho","ma","mi","mu","me","mo","n",
                "na","ni","nu","ne","no","ra","ri","ru","re","ro",
                "ta","chi","tsu","te","to","wa","wo","ya","yo","yu",
                "da", "ji_", "du", "de", "do","za","ji(shi)","zu","ze","zo",
                "ba","bi","bu","be","bo","pa","pi","pu","pe","po", "ga","gi","gu","ge","go"]
lettersN = range(len(letters_half))
trainN = 8

##Read xlsx file
data = pd.read_excel("hog_list.xlsx")
col_name = list(data)

##Get labels
labels = []
for char in col_name:
    c,n = char.split("_")
    labels.append(c)
labels = np.array(labels)


train_labels = np.repeat(lettersN,trainN)[:,np.newaxis]
#print train_labels
print "len train_labels: " + str(len(train_labels))

##train_data
train_data = []
for col in col_name:
    hog = data.get(col)
    hog = np.array(hog)
    train_data.append(hog)
train_data = np.array(train_data)
train_data = train_data.astype(np.float32)
#print train_data
print "len train_data: " + str(len(train_data)) + "   " + str(len(train_data[0]))
print "type of data:" + str(type(train_data[0][0]))

##Set up SVM
SVM_C = 2.67
print cv2.ml.SVM_C_SVC
print cv2.ml.SVM_LINEAR
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(SVM_C)
svm.setGamma(5.1)
print cv2.ml.ROW_SAMPLE

#train
svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
print "Train complete"

###Test
##get test data
testN = 2
data_test = pd.read_excel("hog_list_test.xlsx")
col_test = list(data_test)
test_train_data = []
for col in col_test:
    hog = data_test.get(col)
    hog = np.array(hog)
    test_train_data.append(hog)
test_train_data = np.array(test_train_data)
test_train_data = test_train_data.astype(np.float32)
print "len train_data: " + str(len(test_train_data)) + "   " + str(len(test_train_data[0]))
print "type of data:" + str(type(test_train_data[0][0]))

##predict
test_labels = np.repeat(lettersN,testN)[:,np.newaxis]
#print test_labels

##accuracy
result = svm.predict(test_train_data)
print result[1]
mask = result[1] == test_labels
correct = np.count_nonzero(mask)
accuracy = correct * 100/result[1].size
print "Accuracy = " + str(accuracy)

img_list, lettersN, eachN = readImageList(letters_half,0,8)
img_list = imgProcess(img_list)
hog_list = hog_compute(img_list)
train_labels = labelGen(lettersN,eachN)
print len(train_labels)
print len(hog_list)

svm = svmSetUp()
svmTrain(svm,hog_list,train_labels)
print "Is svm trained: " + str(svm.isTrained())

file_name = "test-data/singles_50x50/a_21.png"
result = svmPredict1Image(svm, file_name, letters_half)
print "Letter is :" + result

accuracyTest(svm, letters_half,8,10)