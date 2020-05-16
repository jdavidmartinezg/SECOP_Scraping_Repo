# -*- coding: utf-8 -*-
"""
Created on Tue May  5 23:25:24 2020

@author: jdavi
"""

import os
import pandas as pd
import datetime as dt

os.chdir(r"D:\SECOP\2020")

'''
RUN ONCE

# Donwload date: 5/5/20 -> last updated on 5th may

secop1 = pd.read_csv("SECOP_I.csv")

secop2 = pd.read_csv("SECOP_II.csv")

# DB first look

secop1_mini = secop1.sample(20)

secop2_mini = secop2.sample(20)


# filter by 2020 only

secop1_2020 = secop1[secop1['Anno Cargue SECOP'] == 2020]

secop2['Fecha de Publicacion del Proceso'] = pd.to_datetime(secop2['Fecha de Publicacion del Proceso'])

secop2['Anno Cargue SECOP'] = secop2['Fecha de Publicacion del Proceso'].dt.year

secop2_2020 = secop2[secop2['Anno Cargue SECOP'] == 2020]

# Save files to pickle

secop1_2020.to_pickle("secop1_2020.pkl")

secop2_2020.to_pickle("secop2_2020.pkl")

'''


# Load pickle datasets

secop1_2020 = pd.read_pickle("secop1_2020.pkl")

secop2_2020 = pd.read_pickle("secop2_2020.pkl")

# Filter by 'Urgencia Manifiesta' type of contracts

pd.crosstab(index = secop1_2020['Causal de Otras Formas de Contratacion Directa'], columns = 'Freq')

pd.crosstab(index = secop2_2020['Justificación Modalidad de Contratación'], columns = 'Freq')

# Filter

# secop1_2020_UM = secop1_2020[secop1_2020['Causal de Otras Formas de Contratacion Directa'].str.contains("Urgencia Manifiesta")]

# secop2_2020_UM = secop2_2020[secop2_2020['Justificación Modalidad de Contratación'].str.contains("Urgencia manifiesta")]


# Diccionario para filtrado de kits de alimentos

contratos_dict = ['kit','aliment','nutrici','comida'] # Mejorar: Pedir ayuda a Jorge

secop1_2020_UM_alimentos = secop1_2020[(secop1_2020['Detalle del Objeto a Contratar'].str.contains('|'.join(contratos_dict), case=False, na=False))|(secop1_2020['Objeto a Contratar']== 'Alimentos, Bebidas y Tabaco')]

secop2_2020_UM_alimentos = secop2_2020[(secop2_2020['Descripción del Procedimiento'].str.contains('|'.join(contratos_dict), case=False, na=False))|(secop2_2020['Nombre del Procedimiento'].str.contains('|'.join(contratos_dict), case=False, na=False))]

# Filtrar por contratación directa

secop1_2020_UM_alimentos = secop1_2020_UM_alimentos[secop1_2020_UM_alimentos['Tipo de Proceso'].str.contains("Contratación Directa", case=False, na=False)]

secop2_2020_UM_alimentos = secop2_2020_UM_alimentos[secop2_2020_UM_alimentos['Modalidad de Contratacion'].str.contains("Contratación Directa", case=False, na=False)]


secop1_2020_UM_alimentos = secop1_2020_UM_alimentos.reset_index(inplace = False)

secop2_2020_UM_alimentos = secop2_2020_UM_alimentos.reset_index(inplace = False)

