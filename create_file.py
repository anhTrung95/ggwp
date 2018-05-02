import cv2

letters_half = ["a","i","u","e","o","ka","ki","ku","ke","ko","sa","shi",
                "su","se","so","ha","hi","fu","he","ho","ma","mi","mu",
                "me","mo","n","na","ni","nu","ne","no","ra","ri","ru","re",
                "ro","ta","chi","tsu","to","te","wa","wo","ya","yo","yu"]

save_folder = 'train_case/'
for letter in letters_half:
    folder = 'test-data/' + letter + '/'
    for i in [0,1]:
        file_name = folder + str(i) + '_50x50.png'
        img = cv2.imread(file_name, 0)
        cv2.imwrite(save_folder + letter + "_20" + str(i) + ".png", img)