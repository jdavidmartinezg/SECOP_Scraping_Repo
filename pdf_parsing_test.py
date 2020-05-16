# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:51:23 2020

@author: jdavi
"""



import camelot # pip install camelot-py[cv]
# Install in clear environment
# Install ghostscript from: https://www.ghostscript.com/download/gsdnld.html
import fitz
import numpy as np
import os
import pandas as pd
import PyPDF4
import re
import io
import time



os.chdir(r"C:\Users\jdavi\Downloads")



# Tables in didigtal document

# Is crutial to specify the page in which the table is in 
# Doo loop over each page to ensure the parsing of table
tables = camelot.read_pdf('C_PROCESO_20-12-10643602_205031011_72747497.pdf',
                          pages='1,2',
                          flavor='lattice') # flavor='lattice'

tables1 = camelot.read_pdf('C_PROCESO_20-12-10658075_205142011_72882086.pdf',
                          pages='2')


# tables = camelot.read_pdf('C_PROCESO_20-12-10643602_205031011_72747497.pdf', pages='2')

tables[0].df
tables1[1].df

tables.export('secop_test1.csv', f='csv', compress=True) 


# Tables in scanned document

import tabula # Works better pip install tabula-py
from tabula import read_pdf

read_pdf('C_PROCESO_20-12-10643602_205031011_72747497.pdf', pages = '1,2')

read_pdf('C_PROCESO_20-12-10658075_205142011_72882086.pdf', pages = '2')

# Scanned table:
    
read_pdf('C_PROCESO_20-12-10633902_205150011_72679995.pdf', pages = '2,3')

tables2 = camelot.read_pdf('C_PROCESO_20-12-10633902_205150011_72679995.pdf',
                          pages='2',
                          flavor='stream')


# OCR Tesseract


from wand.image import Image
# https://imagemagick.org/script/download.php#windows
from PIL import Image as PI
import pyocr
import pyocr.builders
import io
import pytesseract

tool = pyocr.get_available_tools()[0]
# install tesseract: https://github.com/tesseract-ocr/tesseract/wiki#windows

lang = tool.get_available_languages()[2]


req_image = []
final_text = []

image_pdf = Image(filename="C_PROCESO_20-12-10633902_205150011_72679995.pdf", 
                  resolution=300)

image_jpeg = image_pdf.convert('jpeg')


for img in image_jpeg.sequence:
    img_page = Image(image=img)
    req_image.append(img_page.make_blob('jpeg'))

for img in req_image: 
    txt = tool.image_to_string(
        PI.open(io.BytesIO(img)),
        lang=lang,
        builder=pyocr.builders.TextBuilder()
    )
    final_text.append(txt)

final_text