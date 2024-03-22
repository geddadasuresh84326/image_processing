import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imutils
import cv2
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\suresh\AppData\Local\Programs\Tesseract-OCR\tesseract'

folder_path = os.path.join("data")

file_names = os.listdir(folder_path)

dt = {"Company":[],"Name":[],"Mobile":[]}

file_list = []
for file in file_names:
    file_list.append(file)

file_sample = file_list[:5]

print(file_sample)

for file_name in file_sample:
    image = cv2.imread(filename=f"data/{file_name}")

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # plt.imshow(cv2.cvtColor(gray,cv2.COLOR_BGR2RGB))

    # Initialize Tesseract OCR engine
    tesseract_config = r'--oem 3 --psm 6'  # OCR Engine Mode (OEM) 3 for default, Page Segmentation Mode (PSM) 6 for a single uniform block of text
    tesseract_config += r' -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Whitelist of characters
    tesseract_config += r' -c preserve_interword_spaces=1'  # Preserve interword spaces
    tesseract_config += r' -c language_model_penalty_non_dict_word=1'  # Penalize non-dictionary words
    tesseract_config += r' -c language_model_penalty_non_freq_dict_word=1'  # Penalize infrequent dictionary words
    tesseract_config += r' -c language_model_penalty_suffix=1'  # Penalize words based on suffix

    # Perform OCR
    text = pytesseract.image_to_string(gray, config=tesseract_config)

    rows = text.strip().split()

    i = 0
    while(i<len(rows)):
        if (rows[i].isalpha()) and (rows[i+1].isalpha()):
            dt["Company"].append(rows[i])
            dt["Name"].append(rows[i+1])
            dt["Mobile"].append(rows[i+2])
            i +=3
        elif (rows[i].isalpha()) and not (rows[i+1].isalpha()):
            dt["Company"].append(rows[i])
            dt["Name"].append("Null")
            dt["Mobile"].append(rows[i+1])
            i +=2

df = pd.DataFrame(dt)
df.to_csv("sample.csv",index=False)