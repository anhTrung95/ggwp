from hog_and_svm import *
from Tkinter import *
from tkFileDialog import askopenfile
from PIL import ImageTk, Image
import xlsxwriter


class FrameTop:

    def __init__(self, main):
        self.im = Image.open("empty.png")
        self.photo = ImageTk.PhotoImage(self.im)
        self.cv_test = Canvas(main)
        self.cv_test.pack()
        self.im_on_cv = self.cv_test.create_image(50, 50, image=self.photo)

    def canvasChange(self, photo):
        self.cv_test.itemconfig(self.im_on_cv, image = photo)

def openFile():
    filepath = askopenfile(parent=root)
    filename = filepath.name
    im = Image.open(filename)
    photo = ImageTk.PhotoImage(im)
    frameTop.canvasChange(photo)
    result = svmPredict1Image(svm, filename, letters_half)
    re_entry.delete(0, END)
    re_entry.insert(0, result)
    frameTop.update()
    root.update()
    print result

root = Tk()
root.title("SVM")
root.geometry("200x200")
root.resizable(width=False, height=False)
frameBtm = Frame(root)
frameBtm.pack(side=BOTTOM)
re_label = Label(frameBtm, text="result: ")
re_label.grid(row=0)
re_entry = Entry(frameBtm)
re_entry.grid(row=0,column=1)
frameTop = FrameTop(root)

menu = Menu(root)

addFileLabel = Label(menu)
menu.add_command(label="Open", command = openFile)
menu.add_command(label="Exit", command = root.quit)

#img_list, lettersN, eachN = readImageList(letters_half,0,8)
img_list, lettersN, eachN, train_files = readImageListTrain(letters_half)
img_list = imgProcess(img_list)
hog_list = hog_compute(img_list)
train_labels = labelGen(lettersN,eachN)
#print len(train_labels)
#print len(hog_list)

svm = svmSetUp(2.67)
svmTrain(svm,hog_list,train_labels)
print "Is svm trained: " + str(svm.isTrained())

#print "Accuracy: " + str(accuracyTest(svm, letters_half,5, train_files))


#workbook = xlsxwriter.Workbook("svm.xlsx")
#worksheet = workbook.add_worksheet()
#for i, letter in enumerate(letters_half):
#    acc = accuracyOnLetter(svm,letters_half,letter,100, train_files)
#    worksheet.write(1,i,letter)
#    worksheet.write(2,i, acc)
#workbook.close()
#print "finish xlsx"

root.config(menu=menu)
root.mainloop()
