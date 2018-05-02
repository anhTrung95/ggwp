import cv2
import numpy as np
import xlsxwriter

test_image = cv2.imread("test-data/a/9_50x50.png", 0)

winSize = (50,50)
blockSize = (20,20)
blockStride = (10,10)
cellSize = (5,5)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,
                        winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
list_hog = []
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


print type(letters_10)
for letter in letters_half:
    for i in range(8):
        file_name = "test-data/" + letter + "/" + str(i) + ".png"
        #print file_name
        image = cv2.imread(file_name, 0)
        #cv2.imshow("image", image)
        #print type(image)
        image = cv2.resize(image, (50,50))
        image = cv2.medianBlur(image, 5)
        hist = hog.compute(image)
        print "len: " + str(len(hist))
        list_hog.append(hist)

list_hog = np.array(list_hog)
#print list_hog
letter_in_sheet = []
workbook = xlsxwriter.Workbook("hog_list.xlsx")
worksheet = workbook.add_worksheet()
for letter in letters_half:
    for i in range(8):
        letter_in_sheet.append(letter + "_" + str(i))
worksheet.write_row(0,0,letter_in_sheet)

for col, hog_data in enumerate(list_hog):
    for row, data in enumerate(hog_data):
        worksheet.write_column(row+1, col, data)
print data[0]

workbook.close()

####Test data
list_hog_test = []
for letter in letters_half:
    for i in range(8,10):
        file_name = "test-data/" + letter + "/" + str(i) + ".png"
        #print file_name
        #print file_name
        image = cv2.imread(file_name, 0)
        #cv2.imshow("image", image)
        #print type(image)
        image = cv2.resize(image, (50,50))
        image = cv2.medianBlur(image, 5)
        hist = hog.compute(image)
        list_hog_test.append(hist)

list_hog_test = np.array(list_hog_test)
#print list_hog
letter_in_sheet = []
workbook = xlsxwriter.Workbook("hog_list_test.xlsx")
worksheet = workbook.add_worksheet()
for letter in letters_half:
    for i in range(2):
        letter_in_sheet.append(letter + "_" + str(i))
worksheet.write_row(0,0,letter_in_sheet)

for col, hog_data in enumerate(list_hog_test):
    for row, data in enumerate(hog_data):
        worksheet.write_column(row+1, col, data)

workbook.close()