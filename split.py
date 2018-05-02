import cv2

import collections

EXAMPLE = "test-data/split-test-1.png"

img = cv2.imread(EXAMPLE, cv2.IMREAD_GRAYSCALE)
(thresh, im_reverse) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow('thresh', im_reverse)
cv2.waitKey(0)
height, width = im_reverse.shape

#sum of pixels of each row
rowsums = []
for i in range(height):
	rowsums.append(sum(im_reverse[i]))

#line seperate
PIXEL_THRESH = 5
lines = []
start = None
for i, r in enumerate(rowsums):
	if start == None and r <= PIXEL_THRESH:
		continue
	elif start == None:
		start = i
	elif start != None and r <= PIXEL_THRESH:
		#done with this row
		lines.append((im_reverse[start:i], (start, i)))
		start = None

MIN_WIDTH = 10
chars = collections.defaultdict(list)
for li, (row, (row_start, row_end)) in enumerate(lines):
	start = None
	for i in range(width):
		col = row[:, i]
		if start == None and sum(col) == 0:
			continue
		elif start == None:
			start = i
		elif start != None and sum(col) == 0 and i - start > MIN_WIDTH:
			chars[li].append((img[row_start:row_end, :][:, start:i], (start,i)))
			start = None

for li, (row, (rstart, rend)) in enumerate(lines):
	for _, (cstart, cend) in chars.get(li):
		cv2.rectangle(img, (cstart, rstart), (cend, rend), 1)

cv2.imshow('split_result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
