import numpy as np
import cv2
# import pytesseract as pyt
from matplotlib import pyplot as plt
from skimage.measure import regionprops, label
import pickle as pk
from time import time
import matplotlib.pyplot as plt
from typing import List, Set, Tuple
import os

PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

# print (PATH)

COLORS = [np.array([[120, 124, 126]]), \
          np.array([[201, 180, 88]]), \
          np.array([[106, 170, 100]])]
WHITE = np.array([255, 255, 255])

PK_FNAME = 'c_store.pickle'
PK_HANDLER = open(PATH + '/' + PK_FNAME, 'rb')
PK_DICT = pk.load(PK_HANDLER)

# Generate info from screenshot
def read_img(img_path : str) -> List[Set]:

    '''
    Function for generating information from input screenshot
    '''

    assert isinstance(img_path, str), 'INPUT TO READ_IMG IS WRONG'

    info = [set(), set(), set()]
    
    # Load image
    img = cv2.imread(img_path)
    # img = cv2.imread(img_path) # debug
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    m, n, _ = img.shape
    mask = np.zeros((m, n))
    
    # debug
    # plt.imshow(img)
    # plt.show()

    # Get binary mask
    for i in range(m):
        for j in range(n):
            if np.sum(np.absolute(img[i, j] - WHITE)) < 10:
                mask[i, j] = 1
    
    # Process the mask
    mask = mask.astype('int')
    mask_label = label(mask)

    # Get possible regions from the mask
    regions = regionprops(mask_label)

    # Track index
    index = 0

    for region in regions:

        # Analyze each region, filter based on area and shape
        if region.area > 2000: continue
        min_r, min_c, max_r, max_c = region.bbox
        len_x, len_y = max_r - min_r, max_c - min_c
        # if len_x >= 8 * len_y or len_x <= 1 / 8 * len_y: continue

        # Reshape the current region to match with the data collected
        len_max = max(len_x, len_y)
        pad_x, pad_y = len_max - len_x, len_max - len_y

        curr = mask[min_r : max_r, min_c : max_c]
        curr = np.pad(curr, pad_width = [(0, pad_x), (0, pad_y)], mode = 'constant').astype(np.float32)
        curr = cv2.resize(curr, (22, 22), interpolation = cv2.INTER_AREA)
        
        # debug
        # plt.imshow(curr)
        # plt.show()
        # print (len_x, len_y)

        # Find the character in the curr bbox by argmax
        char_score = np.zeros(26)

        for i in range(26):
            char = chr(97 + i)
            value = PK_DICT[char]
            char_score[i] = np.sum(curr == value) / np.sum(curr != value) if np.sum(curr != value) != 0 else 1000
        
        curr_char = chr(97 + np.argmax(char_score))

        # if curr_char == 'c': 
        #     print (curr.shape)
        #     print (curr.astype('int'))
        #     print (PK_DICT['o'])
        #     print (PK_DICT['c'])
        #     print (char_score)

        # Find the color corresponds to the character
        color = img[min_r - 10, min_c - 10]

        # Track the index of this character
        index %= 5

        curr_color = np.zeros(3)
        for i in range(len(COLORS)):
            curr_color[i] = np.sum(np.absolute(color - COLORS[i])) 
        
        curr_color_index = np.argmin(curr_color)
        info[curr_color_index].add((curr_char, index))

        # Add the color and character inforation
        # for i in range(len(COLORS)):
        #     if np.sum(np.absolute(color - COLORS[i])) < 10:
        #         info[i].add((curr_char, index))
        #         break
        
        index += 1
    
    return info

def get_res(info : List[Set]) -> Tuple[List[str], float]:

    '''
    Function for getting potential answers based on the dictionary of info 
    generated in read_img function
    '''

    assert isinstance(info, list) and len(info) == 3, 'INPUT TO GET_RES IS WRONG'

    t1 = time()

    gray, yellow, green = info
    res, res_full = [], []

    # check commonly used words
    with open(PATH + 'words_five.txt', 'rt') as f:
        for line in f: 
            judge = True

            # check gray 
            for c, idx in gray:
                if line[idx] == c: 
                    judge = False
                    break

            if judge:
                # check green
                for c, idx in green:
                    if line[idx] != c: 
                        judge = False
                        break

            if judge:
                # check yellow
                for c, idx in yellow:
                    if c not in line:
                        judge = False
                        break
                    elif line[idx] == c:
                        judge = False
                        break
            if judge: 
                res.append(line[: -1])
    
    t2 = time()
    if len(res) >= 10: return res, t2 - t1

    # check all words if there are not enough words produced by reading the most frequent word list
    if len(res) <= 10: 
        with open(PATH + 'words_five_full.txt', 'rt') as f:
            for line in f: 
                judge = True

                # check gray 
                for c, idx in gray:
                    if line[idx] == c: 
                        judge = False
                        break
                    
                if judge:
                    # check green
                    for c, idx in green:
                        if line[idx] != c: 
                            judge = False
                            break
                if judge:
                    # check yellow
                    for c, idx in yellow:
                        if c not in line:
                            judge = False
                            break
                        elif line[idx] == c:
                            judge = False
                            break
                if judge: 
                    res_full.append(line[: -1])
            
    t3 = time()

    return res_full, t3 - t2

if __name__ == '__main__':
    print ('')