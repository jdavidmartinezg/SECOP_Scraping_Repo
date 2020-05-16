# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:49:43 2020

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
from pdf2image import convert_from_path
import cv2 # OpenCV 
import matplotlib.pyplot as plt
import csv
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
from pdfminer3.layout import LAParams, LTTextBox # slow digital pdf parser
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.converter import TextConverter
from tika import parser # quickest digital pdf parser
import multiprocessing as mp # speedup with multicore processing
from tqdm import tqdm
from io import StringIO
from bs4 import BeautifulSoup
import pickle




# Parse table function pseudo-code

'''
1) Detect whether PDF is image or digital document
    
2) If PDF is digital detect in which page/pages there are tables
    * Detect which tables are worth keeping (Create a dictionary of food items)    
    
3) If PDF is scanned document, detect in which pages there are tables
    * OpenCV to detect lines
    * OCR and dictionary to detect food tables

4) Extract tables from scanned documents
'''


# path = r'D:\SECOP\2020\SECOP_II'

def PDF_Database(path):
    
    '''
    Parameters
    ----------
    path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    folder_dict = {'ID del Proceso':[],
                   'Archivo':[],
                   'AbsolutePath':[]}
    
    for folder in os.listdir(path):
        folder_dict['ID del Proceso'] = folder_dict['ID del Proceso'] + [folder for i in range(len(os.listdir(path+'\\'+folder)))]
        folder_dict['Archivo'] = folder_dict['Archivo'] + os.listdir(path+'\\'+folder)
        folder_dict['AbsolutePath'] = folder_dict['AbsolutePath'] + [path+'\\'+folder+'\\'+file for file in os.listdir(path+'\\'+folder)]
        
    folder_df = pd.DataFrame(folder_dict)        

    return folder_df


def AnalyzePDF(path):
    # This algorithm calculates the percentage of document that is covered by (searchable) text
    try:

        page_num = 0
        text_perc = 0.0
    
        doc = fitz.open(path)
    
        for page in doc:
            page_num = page_num + 1
    
            page_area = abs(page.rect)
            text_area = 0.0
            for b in page.getTextBlocks():
                r = fitz.Rect(b[:4]) # rectangle where block text appears
                text_area = text_area + abs(r)
            text_perc = text_perc + (text_area / page_area)
    
        text_perc = text_perc / page_num
    
        # If the percentage of text is very low, the document is most likely a scanned PDF
        if text_perc < 0.01:
            return "Scanned"
        else:
            return "Digital"

    except:
        return np.nan

# Para pdfs digitales

def PDF_alimentos(path, keywords = ['arroz','aceite','azúcar', 'azucar',
                                    'café', 'cafe','harina', 'atún', 'atun', 
                                    'panela', 'pasta', 'fríjol', 'frijol',
                                    'lenteja', 'chocolate', 'leche']):
    
    #PDFminer3 works better
    
    try:
        String = "|".join(keywords)    
    
        # open the pdf file
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
        
    
        with open(path, 'rb') as fh:
        
            for page in PDFPage.get_pages(fh,
                                          caching=True,
                                          check_extractable=True):
                page_interpreter.process_page(page)
        
            Text = fake_file_handle.getvalue()
        
        # close open handles
        converter.close()
        fake_file_handle.close()
        
        # define keyterms
        ResSearch = re.search(String, Text.lower())
        
        if ResSearch != None:
            return True
        else:
            return False
        
    except:
        return np.nan
          

def PDF_alimentos_tika(path, keywords = ['arroz','aceite','azúcar', 'azucar',
                                    'café', 'cafe','harina', 'atún', 'atun', 
                                    'panela', 'pasta', 'fríjol', 'frijol',
                                    'lenteja', 'chocolate', 'leche']):
    
    #PDFminer3 works better
    
    try:
        String = "|".join(keywords)    
    
        # open the pdf file
        # Parse data from file
        file_data = parser.from_file(path)
        # Get files text content
        Text = file_data['content']
        
        # define keyterms
        ResSearch = re.search(String, Text.lower())
        
        if ResSearch != None:
            return True
        else:
            return False
        
    except:
        return np.nan


#path = "ocr_test.pdf"
def PDF_alimentosOCR(path, keywords = ['arroz','aceite','azúcar', 'azucar',
                                    'café', 'cafe','harina', 'atún', 'atun', 
                                    'panela', 'pasta', 'fríjol', 'frijol',
                                    'lenteja', 'chocolate', 'leche']):

    try:
        String = "|".join(keywords)
        
        images = PDF2Images(pdf_path = path, return_images = True)
        
        Text = ""
        pages = []
        counter = 0
        for image in images:
            page_text = pytesseract.image_to_string(image,lang='spa')
            Text = Text + " " +  page_text
            ResSearch_page = re.search(String, page_text.lower())
            if ResSearch_page != None:
                pages.append(counter)
            else:
                pass
            counter = counter + 1
        
        ResSearch = re.search(String, Text.lower())
        
    
        if ResSearch != None:
            return True
        else:
            return False
    except:
        return np.nan


def PDF_alimentos_pages(path, keywords = ['arroz','aceite','azúcar', 'azucar',
                                    'café', 'cafe','harina', 'atún', 'atun', 
                                    'panela', 'pasta', 'fríjol', 'frijol',
                                    'lenteja', 'chocolate', 'leche']):
    
    String = "|".join(keywords)
    
    try:
    
        def extract_text_by_page(pdf_path):
            with open(pdf_path, 'rb') as fh:
                for page in PDFPage.get_pages(fh, 
                                              caching=True,
                                              check_extractable=True):
                    resource_manager = PDFResourceManager()
                    fake_file_handle = io.StringIO()
                    converter = TextConverter(resource_manager, fake_file_handle)
                    page_interpreter = PDFPageInterpreter(resource_manager, converter)
                    page_interpreter.process_page(page)
                    
                    text = fake_file_handle.getvalue()
                    yield text
            
                    # close open handles
                    converter.close()
                    fake_file_handle.close()
            
        def extract_text(pdf_path):
            counter = 0
            pages = []
            for page in extract_text_by_page(pdf_path):
                ResSearch = re.search(String, page.lower())
                #print("Page", counter)
                #print(ResSearch)
                
                if ResSearch != None:
                    pages.append(counter)
                else:
                    pass
                
                counter = counter + 1
            return pages
        
        pages = extract_text(path)
        
        try:
            return ','.join([str(i+1) for i in pages])
        except:
            return np.nan

    except:
        return np.nan
    
    

def PDF_alimentos_pages_tika(path, keywords = ['arroz','aceite','azúcar', 'azucar',
                                    'café', 'cafe','harina', 'atún', 'atun', 
                                    'panela', 'pasta', 'fríjol', 'frijol',
                                    'lenteja', 'chocolate', 'leche']):
    
    filename = path
    
    String = "|".join(keywords)
    
    try:
    
        pages = []
        
        # Read PDF file
        data = parser.from_file(filename, xmlContent=True)
        xhtml_data = BeautifulSoup(data['content'])
        for i, content in enumerate(xhtml_data.find_all('div', attrs={'class': 'page'})):
            # Parse PDF data using TIKA (xml/html)
            # It's faster and safer to create a new buffer than truncating it
            # https://stackoverflow.com/questions/4330812/how-do-i-clear-a-stringio-object
            _buffer = StringIO()
            _buffer.write(str(content))
            parsed_content = parser.from_buffer(_buffer.getvalue())
        
            # Add pages
            text = parsed_content['content'].strip()
            
            ResSearch = re.search(String, text.lower())
                #print("Page", counter)
                #print(ResSearch)
                
            if ResSearch != None:
                pages.append(i)
            else:
                pass
        
        try:
            return ','.join([str(i+1) for i in pages])
        except:
            return np.nan

    except:
        return np.nan
    
    

def PDF_alimentos_pagesOCR(path, keywords = ['arroz','aceite','azúcar', 'azucar',
                                    'café', 'cafe','harina', 'atún', 'atun', 
                                    'panela', 'pasta', 'fríjol', 'frijol',
                                    'lenteja', 'chocolate', 'leche']):

    
    String = "|".join(keywords)
    
    images = PDF2Images(pdf_path = path, return_images = True)
    
    # Text = ""
    pages = []
    counter = 0
    for image in images:
        page_text = pytesseract.image_to_string(image,lang='spa')
        # Text = Text + " " +  page_text
        ResSearch_page = re.search(String, page_text.lower())
        if ResSearch_page != None:
            pages.append(counter)
        else:
            pass
        counter = counter + 1
    
    # ResSearch = re.search(String, Text.lower())
    

    try:
        return ','.join([str(i+1) for i in pages]) # return whether pdf contains words and in which pages
    except:
        return np.nan




def ExtractTable(path,pages):
    
    try:
        tables = camelot.read_pdf(path,
                              pages=pages,
                              flavor='lattice') 
        df_list = []
        
        for id_df in range(len(tables)-1):
            df_list.append(tables[id_df].df)
        
        return df_list 
    except:
        return np.nan

#..............................................................................
#..............................................................................
#..............................................................................
#..............................................................................
# OCR - Setup Linux environment
#..............................................................................
#..............................................................................
#..............................................................................
#..............................................................................


# https://eihli.github.io/image-table-ocr/pdf_table_extraction_and_ocr.html
# https://github.com/eihli/image-table-ocr
# https://towardsdatascience.com/poppler-on-windows-179af0e50150
# https://github.com/conda-forge/poppler-feedstock

# Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux

# pdf_path = r"D:\OneDrive - Universidad del rosario\Data Science Consultations\SECOP scraping\Data\ocr_test.pdf"
# path = r"D:\OneDrive - Universidad del rosario\Data Science Consultations\SECOP scraping\OCR"

# https://towardsdatascience.com/a-table-detection-cell-recognition-and-text-extraction-algorithm-to-convert-tables-to-excel-files-902edcf289ec


def PDF2Images(pdf_path, path = None, return_images = False):
    images = convert_from_path(pdf_path, 
                        output_folder=path, # folder en donde se guardan las imágenes # None si no quiero guardar archivos
                        fmt = "jpeg", # jpeg es más comprimido y ofrece mayor rapidez
                        use_pdftocairo=True) # el performance es mejor con pdftocairo
    if return_images == True:
        return images

def CropTables(image):
    BLUR_KERNEL_SIZE = (17, 17)
    STD_DEV_X_DIRECTION = 0
    STD_DEV_Y_DIRECTION = 0
    blurred = cv2.GaussianBlur(image, BLUR_KERNEL_SIZE, STD_DEV_X_DIRECTION, STD_DEV_Y_DIRECTION)
    MAX_COLOR_VAL = 255
    BLOCK_SIZE = 15
    SUBTRACT_FROM_MEAN = -2
    
    img_bin = cv2.adaptiveThreshold(
        ~blurred,
        MAX_COLOR_VAL,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE,
        SUBTRACT_FROM_MEAN,
    )
    vertical = horizontal = img_bin.copy()
    SCALE = 5
    image_width, image_height = horizontal.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image_width / SCALE), 1))
    horizontally_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(image_height / SCALE)))
    vertically_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel)
    
    horizontally_dilated = cv2.dilate(horizontally_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    vertically_dilated = cv2.dilate(vertically_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    
    mask = horizontally_dilated + vertically_dilated
    contours, heirarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    MIN_TABLE_AREA = 1e5
    contours = [c for c in contours if cv2.contourArea(c) > MIN_TABLE_AREA]
    perimeter_lengths = [cv2.arcLength(c, True) for c in contours]
    epsilons = [0.1 * p for p in perimeter_lengths]
    approx_polys = [cv2.approxPolyDP(c, e, True) for c, e in zip(contours, epsilons)]
    bounding_rects = [cv2.boundingRect(a) for a in approx_polys]

    # The link where a lot of this code was borrowed from recommends an
    # additional step to check the number of "joints" inside this bounding rectangle.
    # A table should have a lot of intersections. We might have a rectangular image
    # here though which would only have 4 intersections, 1 at each corner.
    # Leaving that step as a future TODO if it is ever necessary.
    images = [image[y:y+h, x:x+w] for x, y, w, h in bounding_rects]
    return images


def SaveTable(image_path, drop_path, drop_name):
    image_filename = image_path
    
    image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    
    tables = CropTables(image)
    
    counter = 0
    
    for table in tables:
    
        cv2.imwrite(drop_path+"\\"+drop_name+"_"+str(counter)+".jpeg", table)
 
        counter = counter + 1
# SaveTable("output_test.PNG", r"C:\Users\jdavi\Downloads", "test1")

#file = "test1_0.jpeg"




#read your file


def SortContours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def Table2Dataframe(image_path, drop_path, drop_name):

    file = image_path
    
    img = cv2.imread(file,0)
    img.shape
    #thresholding the image to a binary image
    thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
    #inverting the image 
    img_bin = 255-img_bin
    
    
    
    cv2.imwrite(drop_path+"\\"+drop_name+"_"+"cv_inverted"+".jpeg",img_bin)
    
    
    
    #Plotting the image to see the output
    #plotting = plt.imshow(img_bin,cmap='gray')
    #plt.show()
    
    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1]//100
    # Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    #Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    
    
    
    
    cv2.imwrite(drop_path+"\\"+drop_name+"_"+"vertical"+".jpeg",vertical_lines)
    
    
    
    
    
    #Plot the generated image
    #plotting = plt.imshow(image_1,cmap='gray')
    #plt.show()
    
    #Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    
    
    
    
    cv2.imwrite(drop_path+"\\"+drop_name+"_"+"horizontal"+".jpeg",horizontal_lines)
    
    
    
    
    
    #Plot the generated image
    #plotting = plt.imshow(image_2,cmap='gray')
    #plt.show()
    
    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    #Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    
    
    
    cv2.imwrite(drop_path+"\\"+drop_name+"_"+"img_vh"+".jpeg", img_vh)
    
    
    
    
    bitxor = cv2.bitwise_xor(img,img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    #Plotting the generated image
    #plotting = plt.imshow(bitnot,cmap='gray')
    #plt.show()
    
    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort all the contours by top to bottom.
    contours, boundingBoxes = SortContours(contours, method="top-to-bottom")
    
    #Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    #Get mean of heights
    mean = np.mean(heights)
    
    #Create list box to store all boxes in  
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w<1000 and h<500):
            image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            box.append([x,y,w,h])
    #plotting = plt.imshow(image,cmap='gray')
    #plt.show()
    
    #Creating two lists to define row and column in which cell is located
    row=[]
    column=[]
    j=0
    #Sorting the boxes to their respective row and column
    for i in range(len(box)):
        if(i==0):
            column.append(box[i])
            previous=box[i]
        else:
            if(box[i][1]<=previous[1]+mean/2):
                column.append(box[i])
                previous=box[i]
                if(i==len(box)-1):
                    row.append(column)
            else:
                row.append(column)
                column=[]
                previous = box[i]
                column.append(box[i])
    #print(column)
    #print(row)
    
    #calculating maximum number of cells
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol
    
    #Retrieving the center of each column
    center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]
    center=np.array(center)
    center.sort()
    
    #Regarding the distance to the columns center, the boxes are arranged in respective order
    finalboxes = []
    for i in range(len(row)):
        lis=[]
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)
    
    
    #from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    outer=[]
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner=''
            if(len(finalboxes[i][j])==0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                    finalimg = bitnot[x:x+h, y:y+w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg,2,2,2,2,   cv2.BORDER_CONSTANT,value=[255,255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel,iterations=1)
                    erosion = cv2.erode(dilation, kernel,iterations=1)
    
                    
                    out = pytesseract.image_to_string(erosion)
                    if(len(out)==0):
                        out = pytesseract.image_to_string(erosion, config='--psm 3')
                    inner = inner +" "+ out
                outer.append(inner)
    
    
    #Creating a dataframe of the generated OCR list
    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
    # print(dataframe)
    
    return dataframe
    # data = dataframe.style.set_properties(align="left")
    
    # data.to_excel("test_output.xlsx")

# y = Table2Dataframe("test1_0.jpeg", r"C:\Users\jdavi\Downloads", "test1")













###############################################################################


os.chdir(r"D:\SECOP\2020")



# secop2 = PDF_Database(r'D:\SECOP\2020\SECOP_II')

# secop2['PDF Type'] = secop2['AbsolutePath'].map(AnalyzePDF)

# secop2.to_pickle("secop2_2020_file_tree.pkl")

# secop2 = pd.read_pickle("secop2_2020_file_tree.pkl")
secop2 = pd.read_pickle("secop2_2020_file_treev2.pkl")
# secop2 = pd.read_pickle("secop2_2020_file_treev3.pkl")


# secop2 = secop2.drop(columns = ['Comida','Pages'])

pd.crosstab(index = secop2['PDF Type'], columns = 'Freq')/len(secop2)*100

# SECOP II: 44% of PDFs are digital and 56% are scanned


# Detect food-related pdfs (Digital) ~ 11 hrs
start = time.time()
mask = secop2['PDF Type'] == 'Digital'
secop2.loc[mask, 'Comida'] = secop2.loc[mask, 'AbsolutePath'].apply(PDF_alimentos)
print('\n\n\nIt took', time.time()-start, 'seconds.')
# secop2.to_pickle("secop2_2020_file_treev2.pkl")
pd.crosstab(index = secop2['Comida'], columns = 'Freq')
# Se encontraron 415 pdfs


# Detect food-related pdfs (Tika function in theory should be faster) ~7 min
# Works better, it detects more pdfs than PDFMiner
tqdm.pandas()
start = time.time()
mask = secop2['PDF Type'] == 'Digital'
secop2.loc[mask, 'Comida_tika'] = secop2.loc[mask, 'AbsolutePath'].progress_apply(PDF_alimentos_tika)
print('\n\n\nIt took', time.time()-start, 'seconds.')
pd.crosstab(index = secop2['Comida_tika'], columns = 'Freq')
# secop2.to_pickle("secop2_2020_file_treev2.pkl")
# Se encontraron 418 pdfs


# Of those pdfs containing food keywords, detect which pages have the food related information ~20 min
tqdm.pandas()
start = time.time()
mask2 = secop2['Comida_tika'] == True
secop2.loc[mask2, 'Pages'] = secop2.loc[mask2, 'AbsolutePath'].progress_apply(PDF_alimentos_pages) 
print('\n\n\nIt took', time.time()-start, 'seconds.')
# secop2.to_pickle("secop2_2020_file_treev2.pkl")


# Of those pdfs containing food keywords... (Tika function in theory should be faster) ~3 min
# Has problems detecting some of the cases PDFMiner works better
tqdm.pandas()
start = time.time()
mask2 = secop2['Comida_tika'] == True
secop2.loc[mask2, 'Pages_tika'] = secop2.loc[mask2, 'AbsolutePath'].progress_apply(PDF_alimentos_pages_tika) 
print('\n\n\nIt took', time.time()-start, 'seconds.')
# secop2.to_pickle("secop2_2020_file_treev2.pkl")

# secop2 = secop2.drop(columns = ['Pages_tika',"Comida"])
# secop2.columns = ['ID del Proceso', 'Archivo', 'AbsolutePath', 'PDF Type', 'Pages','Comida']
# secop2 = secop2[['ID del Proceso', 'Archivo', 'AbsolutePath', 'PDF Type', 'Comida', 'Pages']]
# secop2.to_pickle("secop2_2020_file_treev3.pkl")

# Detect food-related pdfs (Scanned) ~ hrs 
tqdm.pandas()
start = time.time()
mask3 = secop2['PDF Type'] == 'Scanned'
secop2.loc[mask3, 'Comida'] = secop2.loc[mask3, 'AbsolutePath'].progress_apply(PDF_alimentosOCR)
print('\n\n\nIt took', time.time()-start, 'seconds.')
pd.crosstab(index = secop2[secop2['PDF Type'] == "Scanned"]['Comida'], columns = 'Freq')
# secop2.to_pickle("secop2_2020_file_treev2.pkl")











# Extract tables from the digital pdfs and pages found
secop2_table_dict_digital = {}

pdfs_of_interest_digital = list(secop2[(secop2['Comida'] == True)&(secop2['PDF Type'] == "Digital")].index)

# counter = 0

start = time.time() # ~25 min
for i in tqdm(pdfs_of_interest_digital):
    secop2_table_dict_digital[secop2['ID del Proceso'].iloc[i]] = ExtractTable(secop2['AbsolutePath'].iloc[i],
                                                           secop2['Pages'].iloc[i])
    # counter = counter + 1
    # print(counter, "OK")
print('\n\n\nIt took', time.time()-start, 'seconds.')


pickle.dump(secop2_table_dict_digital, open("secop2_table_dict_digital.pkl", "wb"))  

secop2_table_dict_digital = pickle.load(open("secop2_table_dict_digital.pkl", "rb"))







