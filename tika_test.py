# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:46:16 2020

@author: jdavi
"""


from tika import parser
import re
import numpy as np
from multiprocessing import Pool


file = 'C_PROCESO_20-12-10658075_205142011_72882086.pdf'



pool = Pool()
pool.map(PDF_alimentos_tika, paths)

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
