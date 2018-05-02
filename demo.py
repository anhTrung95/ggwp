from hog_and_svm import *
import xlsxwriter

x = [0.5, 1, 2, 2.67, 3.33, 4, 5, 6]

img_train_list, lettersNtrain, eachNtrain, train_files = readImageListTrain(letters_half)
img_train_list = imgProcess(img_train_list)
train_labels = labelGen(lettersNtrain,eachNtrain)

testNumber = 20
img_test_list, lettersNtest, eachNtest = readImageListTest(letters_half, testNumber, train_files)
img_test_list = imgProcess(img_test_list)
test_labels = labelGen(lettersNtest,eachNtest)
workbook = xlsxwriter.Workbook("demo.xlsx")
worksheet = workbook.add_worksheet()
worksheet.write(0,0,"svm_c")
worksheet.write(0,1,"cellSize")
worksheet.write(0,2,"blockSize")
worksheet.write(0,3,"blockStride")
worksheet.write(0,4,"Accuracy")
worksheet.write(0,5,str(testNumber) + " test cases")
row = 1
#print len(train_labels)
#print len(hog_list)
for i in x:
    for cellSize in [(5,5), (10,10)]:
        for blockSize in [(10,10), (20,20)]:
            for blockStride in [(5,5), (10,10)]:
                hog_train_list = hog_compute_opt(img_train_list, cellSize, blockSize,blockStride)
                hog_test_list = hog_compute_opt(img_test_list, cellSize, blockSize,blockStride)
                svm = svmSetUp(i)
                svmTrain(svm,hog_train_list,train_labels)
                print "Is svm trained: " + str(svm.isTrained())
                result = svm.predict(hog_test_list)
                mask = result[1] == test_labels
                correct = 0
                for j in range(len(test_labels)):
                    if test_labels[j] == result[1][j]:
                        correct += 1
                # print "Number of correct: " + str(correct)
                accuracy = float(correct * 100) / result[1].size
                # print "Accuracy = " + str(accuracy)
                print "Accuracy if svm_c="+ str(i) + ", cellSize=" + str(cellSize) + ", blockSize=" + str(blockSize) +\
                      ", blockStride=" + str(blockStride) +" is " + str(accuracy)
                svm = None
                worksheet.write(row, 0, i)
                worksheet.write(row, 1, str(cellSize[0]) + "," + str(cellSize[1]))
                worksheet.write(row, 2, str(blockSize[0]) + "," + str(blockSize[1]))
                worksheet.write(row, 3, str(blockStride[0]) + "," + str(blockStride[1]))
                worksheet.write(row, 4, accuracy)
                row+=1
workbook.close()
