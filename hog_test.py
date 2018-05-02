import numpy as np
import xlsxwriter
import pandas as pd
import math

def euclide_distance(array_1, array_2):
    if len(array_1) != len(array_2):
        return None
    d = 0
    for i in range(len(array_1)):
        d += (array_1[i] - array_2[i])**2
    return math.sqrt(d) / len(array_1)

df = pd.read_excel("hog_list.xlsx", sheet_name="Sheet1")
df.as_matrix()
print type(df)
print type(df.get("a_0"))
print len(df.get("a_0"))

letters = ["a", "i", "u", "e", "o", "ha", "hi", "fu", "he", "ho"]
letter_in_sheet = []
for j, letter in enumerate(letters):
    for i in range(8):
        letter_in_sheet.append(letter + "_" + str(i))

ed_list = []
for img, data in enumerate(letter_in_sheet):
    if img % 8 != 0:
        continue
    ed_1_letter = []
    for letter in letter_in_sheet:
        ed_1_letter.append(euclide_distance(df.get(data), df.get(letter)))
    ed_1_letter = np.array(ed_1_letter)
    ed_list.append(ed_1_letter)

ed_list = np.array(ed_list)
#print ed_list

workbook = xlsxwriter.Workbook("euclide_distance.xlsx")
worksheet = workbook.add_worksheet()

for i, data in enumerate(ed_list):
    worksheet.write_column(0, i, data)
workbook.close()