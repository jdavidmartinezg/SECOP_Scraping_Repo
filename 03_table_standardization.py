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
import re
import statistics as sta
import price_parser as pp
import itertools
import datetime as dt
from pandas import ExcelWriter
from pandas import ExcelFile

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

def CleanExtraWhiteSpaces(s):
    return re.sub(' {2,}', " ", s.strip()) # Limpiar espacios extra en los strings

def CleanDFs(df_list):

    clean_df_list = []
    for df in df_list:
        n_df = df.applymap(CleanExtraWhiteSpaces)
        n_df = n_df.replace('', np.nan)
        n_df = n_df.replace(' ', np.nan)
        n_df = n_df.dropna(how='all', axis=0) # filas vacías
        n_df = n_df.dropna(how='all', axis=1) # columnas vacías
        clean_df_list.append(n_df)

    return clean_df_list

secop2_table_dict_digital_food_prices = valmap(CleanDFs, secop2_table_dict_digital_food_prices)
secop2_table_dict_scanned_food_prices = valmap(CleanDFs, secop2_table_dict_scanned_food_prices)


# Limpieza de tablas cuya 'sparsity' sea muy alta: Threshold 70%

def TableSparsityFilter(df_list, threshold = 0.75):

    clean_df_list = []

    for df in df_list:

        percent_missing = np.sum(df.isnull().sum())/(len(df)*len(test_df.columns))

        if percent_missing < threshold:

            clean_df_list.append(df)

    return clean_df_list


secop2_table_dict_digital_food_prices = valmap(TableSparsityFilter, secop2_table_dict_digital_food_prices)
secop2_table_dict_scanned_food_prices = valmap(TableSparsityFilter, secop2_table_dict_scanned_food_prices)


# Volver a limpiar contratos sin objetos (esta vez por falta de precios)

secop2_table_dict_digital_food_prices = {k: v for k, v in secop2_table_dict_digital_food_prices.items() if len(v) > 0}
secop2_table_dict_scanned_food_prices = {k: v for k, v in secop2_table_dict_scanned_food_prices.items() if len(v) > 0}





# Limpieza de tablas que solo tengan una fila o una columna

def OneRowTableFilter(df_list):

    clean_df_list = []

    for df in df_list:

        if len(df) > 1:

            if len(df.columns) > 1:

                clean_df_list.append(df)

    return clean_df_list


secop2_table_dict_digital_food_prices = valmap(OneRowTableFilter, secop2_table_dict_digital_food_prices)
secop2_table_dict_scanned_food_prices = valmap(OneRowTableFilter, secop2_table_dict_scanned_food_prices)


# Volver a limpiar contratos sin objetos (esta vez por falta de precios)

secop2_table_dict_digital_food_prices = {k: v for k, v in secop2_table_dict_digital_food_prices.items() if len(v) > 0}
secop2_table_dict_scanned_food_prices = {k: v for k, v in secop2_table_dict_scanned_food_prices.items() if len(v) > 0}





# Limpieza de columnas y filas en tablas que tengan más del 60% de values missing

def FilterMissingsColumnsRows(df_list, threshold = 0.6):
    
    clean_df_list = []
       
    for df in df_list:
        columns2drop = []
        rows2drop = []
        for column in df.columns:
            missing_rate_c = df[column].isnull().sum()/len(df[column])
            if missing_rate_c > threshold:
                columns2drop.append(column)
                
        df = df.drop(columns = columns2drop)        
                
        for index in df.index:
            missing_rate_r = df.loc[index].isnull().sum()/len(df.loc[index])
            if missing_rate_r > threshold:
                rows2drop.append(index)            
            
        df = df.drop(index = rows2drop)
        
        clean_df_list.append(df)

    return clean_df_list
    
secop2_table_dict_digital_food_prices = valmap(FilterMissingsColumnsRows, secop2_table_dict_digital_food_prices)
secop2_table_dict_scanned_food_prices = valmap(FilterMissingsColumnsRows, secop2_table_dict_scanned_food_prices)

# Volver a limpiar contratos sin objetos (esta vez por falta de precios)

secop2_table_dict_digital_food_prices = {k: v for k, v in secop2_table_dict_digital_food_prices.items() if len(v) > 0}
secop2_table_dict_scanned_food_prices = {k: v for k, v in secop2_table_dict_scanned_food_prices.items() if len(v) > 0}







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

# sns.distplot(nchar_list)
# sns.boxplot(y=nchar_list, showfliers=False)

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
    try:
        return dotprod / (magA * magB)
    except:
        return 0

def ListsSim(list1, list2):
     commonTerms = set(list1).intersection(set(list2))
     counter = Counter(list2)
     score = sum((counter.get(term) for term in commonTerms)) #edited
     return score


def DigitsPct(df_column): # A nivel de palabra
    word_count = 0
    digit_count = 0
    for item in df_column:
        if item != np.nan:
            word_count = word_count + len(str(item).split())
            # digit_count = digit_count + len(re.findall("(?=(\d{3}))",str(item))) # 3 Digitos consecutivos, para evitar que números pequeños sean catalogados como precios
            # digit_count = digit_count + len(re.findall("\d+((,\d+)+)?(.\d+)?(.\d+)?(,\d+)?",str(item))) # Regex especial para precios
            digit_count = digit_count + len(re.findall("\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})",str(item)))
        else:
            pass
    try:
        return round((digit_count/word_count)*100,2)
    except:
        return 0
    
def DigitsPct2(df_column): # A nivel de caracter
    char_count = 0
    digit_count = 0
    for item in df_column:
        if item != np.nan:
            char_count = char_count + len(re.sub(" ", "", str(item))) # Número de caracteres sin tener en cuenta espacios
            digit_count = digit_count + len(re.findall("\d",str(item))) 
        else:
            pass
    try:
        return round((digit_count/char_count)*100,2)
    except:
        return 0
    
    
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

    # counterA = Counter(food_keywords)
    # counterB = Counter([words for segments in list(df_column) for words in str(segments).split()])

    # return round(counter_cosine_similarity(counterA, counterB) * 100,2)
    
    foods_regex = ""
    
    for keyword in food_keywords:
        foods_regex = foods_regex + "(?:\\b{term}\\b)|".format(term = keyword)
    
    foods_regex = foods_regex[:-1] # Eliminar último |     
    
    num_matches = len(df_column.str.findall(foods_regex).loc[pd.notna(df_column.str.findall(foods_regex))].sum())
    

    try:

        return round(num_matches/len(df_column.str.split().sum())*100,2) # número de matches sobre número de palabras en columna

    except:
        
        return 0.00
    

# https://regex101.com/
def UnitsSim(df_column, units_keywords = [
                            "lb",
                            "lbr",
                            "libra",
                            "libras",
                            "lbra",
                            "lbras",
                            "ml",
                            "mililitro",
                            "mililitros",
                            "gr",
                            "grs",
                            "gramo",
                            "gramos",
                            "gm",
                            "gms",
                            "g",
                            "kg",
                            "kilo",
                            "kilogramo",
                            "kilos",
                            "kilogramos",
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
                            "cc",
                            "frasco",
                            "sobre",
                            "sbr",
                            "sobres",
                            "porcion",
                            "porciones",
                            "porción"
                            ]):

    
    units_regex = ""
    
    for keyword in units_keywords:
        units_regex = units_regex + "(?:\\b{term}\s\d+)|(?:\\b{term}\\b)|(?:\d+\s{term}\\b)|(?:\\b{term}\d+)|(?:\d+{term}\\b)".format(term = keyword) + "|" 
        # units_regex = units_regex + "(?:\\b{term}\s((?:\d+.\d+)|(?:\d+,\d+)|(?:\d+)))|(?:\\b{term}\\b)|(?:((?:\d+.\d+)|(?:\d+,\d+)|(?:\d+))\s{term}\\b)|(?:\\b{term}((?:\d+.\d+)|(?:\d+,\d+)|(?:\d+)))|(?:((?:\d+.\d+)|(?:\d+,\d+)|(?:\d+)){term}\\b)".format(term = keyword) + "|" 
        # regex que tiene en cuenta decimales, usar: list(itertools.chain.from_iterable(list2d))
    
    
    units_regex = units_regex[:-1] # Eliminar último | 
    
    num_matches = len(df_column.str.findall(units_regex).loc[pd.notna(df_column.str.findall(units_regex))].sum())

    return round(num_matches/len(df_column)*100,2) # número de matches sobre número de items en columna


def EstimateColumnsType(df):

    # Parámetros:

    # porcentaje de dígitos sobre el total de caracteres -> DigitsPct
    # porcentaje de similiradidad con diccionario de comida -> FoodSim
    # procentaje de similaridad con diccionario de unidades -> UnitsSim

    scores = {}

    for col_num in range(len(df.columns)):

        scores[col_num] = {}

        scores[col_num]['prices'] = DigitsPct(df.iloc[:,col_num])
        #scores[col_num]['prices'] = DigitsPct2(df.iloc[:,col_num])
        scores[col_num]['food'] = FoodSim(df.iloc[:,col_num])
        scores[col_num]['units'] = UnitsSim(df.iloc[:,col_num])

    return scores

#test_df2 = secop2_table_dict_focus['CO1.REQ.1262130'][5]
#test_df3 = secop2_table_dict_focus['CO1.REQ.1229650'][5]
#df_column = test_df3.iloc[:,0]

#EstimateColumnsType(test_df3)

# Calcular la desviación estándar de los scores
# Columnas con desviación estándar baja son catalogadas como las 2 columnas con mayor score
# Columnas con desviación estándar alta, se asignan al tipo con mayor score


def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

def getKeysByValues(dictOfElements, listOfValues):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] in listOfValues:
            listOfKeys.append(item[0])
    return  listOfKeys


def NameColumns(df, sd_threshold = 5): # Calibrar parámetro sd_threshold

    scores = EstimateColumnsType(df)
    names = []

    for col_num in range(max(scores.keys())+1):
        sd = sta.stdev(list(scores[col_num].values()))
        if (sd < sd_threshold) & (sd > 0) & (list(scores[col_num].values()).count(0)<2):
            sorted_scores = sorted(list(scores[col_num].values()), reverse=True)
            max2 = sorted_scores[0:2]
            name = "-".join(getKeysByValues(scores[col_num],max2))
        elif sd == 0:
            name = "unknown:"+str(col_num)
        else:
            name = "-".join(getKeysByValue(scores[col_num],max(scores[col_num].values())))

        names.append(name)

    df.columns = names

    return df

# Pueden haber múltiples columnas de precios, unidades y comida. Una opción es dejar sólo una (la que tenga el score más alto de todos las columnas) pero se podría perder información

# https://github.com/scrapinghub/price-parser



# Crear diccionario con contratos, un subdiccionario para cada tabla con un objeto que contenga el dataframe y otro con los scores

def NameColumnsDFList(df_list):

    renamed_df_list = []

    for df in df_list:

        renamed_df = NameColumns(df)

        renamed_df_list.append(renamed_df)

    return renamed_df_list


secop2_table_dict_focus_named = valmap(NameColumnsDFList, secop2_table_dict_focus)



# calibrar funciones, muchos falsos positivos

# Test cases para precios:
    
prices_test_cases = pd.read_csv('prices_test_cases.txt', sep="\n", header=None)    
prices_test_cases.columns = ['test_case']
prices_test_cases['test_case'] = prices_test_cases['test_case'].str.strip()
prices_test_cases['nchar'] = prices_test_cases['test_case'].map(len)
prices_test_cases['type'] = prices_test_cases['test_case'].apply(lambda x: re.sub("\d","",x))
prices_test_cases['group'] = prices_test_cases['nchar'].astype(str) + prices_test_cases['type']
prices_test_cases = prices_test_cases.sort_values(['nchar'], ascending = False)

writer = pd.ExcelWriter('prices_test_cases_sorted.xlsx', engine='xlsxwriter')
prices_test_cases.to_excel(writer, sheet_name='Ex1')
writer.save()
# Use RegexMagic 


# Para cada dataset en cada contrato se iteran todas las filas buscando productos, unidades, cantidades y precios

def ScanTableDict(table_dict, 
                  food_keywords = [
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
                            'leche',
                            'sal'
                            ],
                  units_keywords = [
                            "lb",
                            "lbr",
                            "libra",
                            "libras",
                            "lbra",
                            "lbras",
                            "ml",
                            "mililitro",
                            "mililitros",
                            "gr",
                            "grs",
                            "gramo",
                            "gramos",
                            "gm",
                            "gms",
                            "g",
                            "kg",
                            "kilo",
                            "kilogramo",
                            "kilos",
                            "kilogramos",
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
                            "cc",
                            "frasco",
                            "sobre",
                            "sbr",
                            "sobres",
                            "porcion",
                            "porciones",
                            "porción"
                            ]):
    # table_dict = secop2_table_dict_focus_named
    # df = test_df2
    # row_num = 0
    secop2_products_per_contract = {}
    
    units_regex = ""
    
    for keyword in units_keywords:
        units_regex = units_regex + "(?:\d+\s{term}\\b)|(?:\d+{term}\\b)".format(term = keyword) + "|" 
        # units_regex = units_regex + "(?:\\b{term}\s((?:\d+.\d+)|(?:\d+,\d+)|(?:\d+)))|(?:\\b{term}\\b)|(?:((?:\d+.\d+)|(?:\d+,\d+)|(?:\d+))\s{term}\\b)|(?:\\b{term}((?:\d+.\d+)|(?:\d+,\d+)|(?:\d+)))|(?:((?:\d+.\d+)|(?:\d+,\d+)|(?:\d+)){term}\\b)".format(term = keyword) + "|" 
        # regex que tiene en cuenta decimales, usar: list(itertools.chain.from_iterable(list2d))
        
    units_regex = units_regex[:-1] # Eliminar último |
    
    for contract in table_dict.keys():
        secop2_products_per_contract[contract] = {}
        for df in table_dict[contract]:
            df['combined'] = df[df.columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            for row_num in range(len(df)):
                try:
                    product = list(set(re.findall("|".join(food_keywords), df.iloc[row_num,-1])))[0]
                except:
                    product = "Unkown"
                # price = list(set(re.findall("\d+((,\d+)+)?(.\d+)?(.\d+)?(,\d+)?", df.iloc[row_num,-1])))
                # prices = list(set(re.findall("\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})", df.iloc[row_num,-1])))
                prices = list(set(re.findall("\$ [0-9]\.[0-9]0{2}|\$ [0-9]\.[0-9]{2}0\.0{2}|\$ [0-9]0[0-9].0{2}|\$ [0-9]{2}\.[0-9]{2}0\.0{2}|\$80{2}|\$[0-9]{2}0{3}\.0{2}|8[0-9]0|[0-9]\.[0-9]{2}0|.{3}[0-9]0{2}", df.iloc[row_num,-1])))
                try:
                    price = [price for price in prices if len(price) == min([len(price) for price in prices])][0] # dejar precio mínimo en términos de caracteres
                except:
                    price = "Unkown"
                # price = pp.Price.fromstring(df.iloc[row_num,-1])
                try:
                    units = list(set(re.findall(units_regex, df.iloc[row_num,-1])))[0]
                except:
                    units = "Unkown"
                    
                quantitites = list(set(re.findall("\\b\d+\\b", df.iloc[row_num,-1])))
                quantities_chr = [quantity for quantity in quantitites if len(quantity) == min([len(quantity) for quantity in quantitites])]
                try:
                    quantity = min([int(q) for q in quantities_chr])
                except:
                    quantity = "Unkown"
                
                if (product != "Unkown") and (price != "Unkown") and (units != "Unkown") and (quantity != "Unkown"):
                    
                    if product not in secop2_products_per_contract[contract].keys():
                        secop2_products_per_contract[contract][product] = {"price":price,
                                                                       "units":units,
                                                                       "quantity":quantity}
                    
    return secop2_products_per_contract



secop2_products_per_contract = ScanTableDict(secop2_table_dict_focus_named)

secop2_products_per_contract_df = pd.DataFrame.from_dict({(i,j): secop2_products_per_contract[i][j] 
                           for i in secop2_products_per_contract.keys() 
                           for j in secop2_products_per_contract[i].keys()},
                       orient='index')

secop2_products_per_contract_df = secop2_products_per_contract_df.reset_index()

secop2_products_per_contract_df.columns = ['ID del Proceso', 'Producto', 'Precio', 'Unidades', 'Cantidad']

# Pegar información de fecha, semana del 2020, municipio

# cargar módulo 02_db_filter

secop2_extra_info = secop2_2020_UM_alimentos[['ID del Proceso','Fecha de Publicacion del Proceso', 'Ciudad de la Unidad de Contratación']]

secop2_extra_info.columns = ['ID del Proceso','Fecha', 'Municipio']

secop2_extra_info['Fecha'] = pd.to_datetime(secop2_extra_info['Fecha'])

secop2_extra_info['Semana del 2020'] = secop2_extra_info['Fecha'].dt.week



secop2_products_per_contract_df = pd.merge(
    secop2_products_per_contract_df,
    secop2_extra_info,
    on = "ID del Proceso",
    how = 'left')














