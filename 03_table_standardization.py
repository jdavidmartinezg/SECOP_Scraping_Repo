# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:35:02 2020

@author: jdavi
"""

import pandas as pd
import numpy as np
from toolz.dicttoolz import valmap
import pickle
import os
from collections import Counter
import math

os.chdir(r"D:\OneDrive - Universidad del rosario\Data Science Consultations\SECOP scraping\Data\Python Objects")


# Load dictionary objects

secop2_table_dict_digital = pickle.load(open("secop2_table_dict_digital.pkl", "rb"))
secop2_table_dict_scanned = pickle.load(open("secop2_table_dict_scanned.pkl", "rb"))    


# Mantener contratos que tienen objetos

secop2_table_dict_scanned_notempty = {k: v for k, v in secop2_table_dict_scanned.items() if len(v) > 0}


# Para archivos tipo digital, eliminar nan:
    
secop2_table_dict_digital_notempty = {k: v for k, v in secop2_table_dict_digital.items() if type(v) is not float}
    
secop2_table_dict_digital_notempty = {k: v for k, v in secop2_table_dict_digital_notempty.items() if len(v) > 0}

# Test df

test_df = secop2_table_dict_digital_notempty['CO1.REQ.1089585'][1]
test_df = pd.concat([test_df[col].astype(str).str.lower() for col in test_df.columns], axis=1)
test_list = secop2_table_dict_digital_notempty['CO1.REQ.1089585']
test_list.append(pd.DataFrame({'jkah':[4,4,4]}))


# Convertir todo el texto en minúscula

def dfs2lower(df_list):
    new_list = []
    for df in df_list:
        new_list.append(pd.concat([df[col].astype(str).str.lower() for col in df.columns], axis=1))
    return new_list


secop2_table_dict_digital_notempty = valmap(dfs2lower, secop2_table_dict_digital_notempty)

secop2_table_dict_scanned_notempty = valmap(dfs2lower, secop2_table_dict_scanned_notempty)




# Filtro de tablas que contienen palabras clave:
    
def FilterDfs(df_list, keywords = ['arroz','aceite','azúcar', 'azucar', 'café', 'harina', 'atún', 'atun', 
                                    'panela', 'pasta', 'fríjol', 'frijol',
                                    'lenteja', 'chocolate', 'leche']):
    
    keywords = "|".join(keywords)
    
    filtered_df_list = []    
    for df in df_list:
        match_sum = 0
        for col in df.columns:
            match_sum = match_sum + np.sum(df[col].astype(str).str.contains(keywords))
        if match_sum > 0:
            filtered_df_list.append(df)
    return filtered_df_list

# x = FilterDfs(test_list)

secop2_table_dict_digital_notempty_filtered = valmap(FilterDfs, secop2_table_dict_digital_notempty)

secop2_table_dict_scanned_notempty_filtered = valmap(FilterDfs, secop2_table_dict_scanned_notempty)


# Volver a limpiar contratos sin objetos (esta vez por falta de palabras clave)

secop2_table_dict_digital_food = {k: v for k, v in secop2_table_dict_digital_notempty_filtered.items() if len(v) > 0}
secop2_table_dict_scanned_food = {k: v for k, v in secop2_table_dict_scanned_notempty_filtered.items() if len(v) > 0}


# Filtro de tablas que contengan precios (números):

def FilterDfsPrices(df_list):
    
    filtered_df_list = []    
    for df in df_list:
        match_sum = 0
        for col in df.columns:
            match_sum = match_sum + np.sum(df[col].astype(str).str.contains("\d"))
        if match_sum > 0:
            filtered_df_list.append(df)
    return filtered_df_list

secop2_table_dict_digital_food_prices = valmap(FilterDfsPrices, secop2_table_dict_digital_food)
secop2_table_dict_scanned_food_prices = valmap(FilterDfsPrices, secop2_table_dict_scanned_food)

# Volver a limpiar contratos sin objetos (esta vez por falta de precios)

secop2_table_dict_digital_food_prices = {k: v for k, v in secop2_table_dict_digital_food_prices.items() if len(v) > 0}
secop2_table_dict_scanned_food_prices = {k: v for k, v in secop2_table_dict_scanned_food_prices.items() if len(v) > 0}


# Limpieza de tablas: quitar filas y columnas vacías

def CleanDFs(df_list):

    clean_df_list = []
    for df in df_list:
        n_df = df.replace('', np.nan)
        n_df = n_df.dropna(how='all', axis=0) # filas vacías
        n_df = n_df.dropna(how='all', axis=1) # columnas vacías
        clean_df_list.append(n_df)
        
    return clean_df_list
        
secop2_table_dict_digital_food_prices = valmap(CleanDFs, secop2_table_dict_digital_food_prices)
secop2_table_dict_scanned_food_prices = valmap(CleanDFs, secop2_table_dict_scanned_food_prices)    


# Limpieza de tablas cuya 'sparsity' sea muy alta: Threshold 70% 

def TableSparsityFilter(df_list, threshold = 0.7):

    clean_df_list = []

    for df in df_list:

        percent_missing = np.sum(df.isnull().sum())/(len(df)*len(test_df.columns))

        if percent_missing < threshold:
            
            clean_df_list.append(df)

    return clean_df_list


secop2_table_dict_digital_food_prices = valmap(TableSparsityFilter, secop2_table_dict_digital_food_prices)
secop2_table_dict_scanned_food_prices = valmap(TableSparsityFilter, secop2_table_dict_scanned_food_prices) 


# Combinar fuentes: Digital + Scanned

secop2_table_dict = {key:secop2_table_dict_digital_food_prices.get(key,[])+secop2_table_dict_scanned_food_prices.get(key,[]) for key in set(list(secop2_table_dict_digital_food_prices.keys())+list(secop2_table_dict_scanned_food_prices.keys()))}
# https://stackoverflow.com/questions/26910708/merging-dictionary-value-lists-in-python

def CountTables(d):
    tables = 0
    for key in d.keys():
        tables = tables + len(d[key])

    return tables

#Revisar que el número total de tablas se preserve

CountTables(secop2_table_dict)==(CountTables(secop2_table_dict_digital_food_prices) + CountTables(secop2_table_dict_scanned_food_prices))


# Detectar celdas en tablas que contengan un número de caracteres inusual

# Hacer histograma de número de caracteres por celda
nchar_list = []

for contract in secop2_table_dict.keys():
    for table in secop2_table_dict[contract]:
        for row_num in range(len(table)):
            for col_num in range(len(table.columns)):
                try:
                    nchar_list.append(len(table.iloc[row_num,col_num]))
                except:
                    nchar_list.append(0)

sns.distplot(nchar_list)
sns.boxplot(y=nchar_list, showfliers=False)

np.percentile(np.array(nchar_list), 90)
# 90% de las celdas tienen 24 caracteres o menos

# 'estos son 24 caracteres'


#..............................................................................
#..............................................................................
#..............................................................................
#..............................................................................
#..........................  ANALISIS FOCALIZADO  .............................
#..............................................................................
#..............................................................................
#..............................................................................
#..............................................................................








# 23 contratos para revisar

contracts_of_interest = [
"1207038",
"1218820",
"1228602",
"1228925",
"1229650",
"1242757",
"1252826",
"1254768",
"1257583",
"1259421",
"1266106",
"1267062",
"1267292",
"1145907",
"1242757",
"1243512",
"1244368",
"1244714",
"1250964",
"1252826",
"1256854",
"1258115",
"1262130",
"1262454",
"1262541",
"1266106"
]
    
contracts_of_interest = ["CO1.REQ."+id for id in contracts_of_interest]    
contracts_of_interest = list(set(contracts_of_interest)) # Remover duplicados


secop2_table_dict_focus = {key: secop2_table_dict[key] for key in contracts_of_interest}







# Estimación de tipo de columna

# Tipos: Producto, Unidades, Producto-Unidades, Precio

def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = np.sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(np.sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(np.sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

def ListsSim(list1, list2):
     commonTerms = set(list1).intersection(set(list2))
     counter = Counter(list2)
     score = sum((counter.get(term) for term in commonTerms)) #edited
     return score


def DigitsPct(df_column):
    chr_count = 0
    digit_count = 0
    for item in df_column:
        if item != np.nan:
            chr_count = chr_count + len(item)
            digit_count = digit_count + len(re.findall("\d",item))
        else:
            pass
    return round((digit_count/chr_count)*100,2)

def FoodSim(df_column, food_keywords = [
                            'arroz',
                            'aceite',
                            'azúcar', 
                            'azucar', 
                            'café', 
                            'harina', 
                            'atún', 
                            'atun', 
                            'panela', 
                            'pasta', 
                            'fríjol', 
                            'frijol',
                            'lenteja', 
                            'chocolate', 
                            'leche'
                            ]):
 
    counterA = Counter(food_keywords)
    counterB = Counter([words for segments in list(df_column) for words in segments.split()])
        
    
    return round(counter_cosine_similarity(counterA, counterB) * 100,2)
    
    
def UnitsSim(df_column, untis_keywords = [
                            "lb",
                            "lbr",
                            "libra",
                            "libras",
                            "ml",
                            "mililitro",
                            "mililitros",
                            "gr",
                            "grs",
                            "gramo",
                            "gramos",
                            "g",
                            "kg",
                            "kilo",
                            "kilogramo",
                            "kilos",
                            "kilogramos"
                            "bolsa",
                            "bolsas",
                            "lata",
                            "latas",
                            "paquete",
                            "paquetes",
                            "unidad",
                            "unid",
                            "unds",
                            "unidades",
                            "botella",
                            "botellas",
                            "empaque",
                            "empaques",
                            "oz",
                            "onzas",
                            "onz",
                            "cc"
                            ]):   
    
    counterA = Counter(untis_keywords)
    counterB = Counter([words for segments in list(df_column) for words in segments.split()])
        
    
    return round(counter_cosine_similarity(counterA, counterB) * 100,2)
    

def EstimateColumnsType(df):
    
    # Parámetros: 
        
    # porcentaje de dígitos sobre el total de caracteres -> DigitsPct
    # porcentaje de similiradidad con diccionario de comida -> 
    # procentaje de similaridad con diccionario de unidades ->
    
    scores = {}
    
    for col_num in range(len(df.columns)):
        
        scores[col_num] = {}
        
        scores[col_num]['prices'] = DigitsPct(df.iloc[:,col_num])
        scores[col_num]['food'] = FoodSim(df.iloc[:,col_num])   
        scores[col_num]['units'] = UnitsSim(df.iloc[:,col_num])

    return scores

test_df2 = secop2_table_dict_focus['CO1.REQ.1262130'][5]

EstimateColumnsType(test_df2)





































