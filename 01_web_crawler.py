# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:00:27 2020

@author: jdavi
"""

import selenium
from bs4 import BeautifulSoup
import scrapy
import time
import random



def RandomPause(min_wait = 2, max_wait = 5):
    time.sleep(random.randint(0,max_wait))
    return


# SECOP 1

# Geckodriver

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options

# Set up the driver.

profile = webdriver.FirefoxProfile()

# Cookies
profile.set_preference('browser.cache.disk.enable', False)
profile.set_preference('browser.cache.memory.enable', False)
profile.set_preference('browser.cache.offline.enable', False)
profile.set_preference('network.cookie.cookieBehavior', 2)
# profile.set_preference("browser.privatebrowsing.autostart", True)
profile.set_preference("security.fileuri.strict_origin_policy", False)
profile.update_preferences()


# Capabilities
capabilities = webdriver.DesiredCapabilities.FIREFOX
capabilities['marionette'] = True


firefox_binary = r'C:\bin\gecko\bin\geckodriver.exe'  
driver = webdriver.Firefox(executable_path=firefox_binary, 
                           firefox_profile=profile,
                           capabilities = capabilities)

# extension_path = r'C:\Users\jdavi\webdrivers\buster_captcha_solver_for_humans-0.6.0.xpi' # More stable version
extension_path = r'C:\Users\jdavi\webdrivers\buster_captcha_solver_for_humans-0.7.1-an+fx.xpi'  # Latest version
driver.install_addon(extension_path, temporary=True)

driver.delete_all_cookies()

driver.get('https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=20-12-10683742')
WebDriverWait(driver, 10).until(EC.frame_to_be_available_and_switch_to_it((By.CSS_SELECTOR,"iframe[name^='a-'][src^='https://www.google.com/recaptcha/api2/anchor?']")))
WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//span[@id='recaptcha-anchor']"))).click()
driver.switch_to.default_content()
RandomPause(5)
try:
    all_frames = driver.find_elements_by_tag_name('iframe')
    driver.switch_to.frame(all_frames[4]) # reCaptcha está en frame 4
    RandomPause(5)
    driver.find_element_by_id('solver-button').click()
    time.sleep(20)
    driver.switch_to.default_content()
    driver.find_element_by_xpath("//input[@type='submit' and @value='Enviar']").click()
    
    # while reCaptcha error appears try again refreshing and refresihng
    
# hacer caso para cuando no aparezca reCaptcha
except:
    driver.switch_to.default_content()
    driver.find_element_by_xpath("//input[@type='submit' and @value='Enviar']").click()

# Resolver: No se puede conectar con reCaptcha
# No se puede conectar con reCAPTCHA. Comprueba tu conexión e inténtalo de nuevo. -> delete adblocker (Avast)
# Change browser to chrome

'''
"We're sorry, but your computer or network may be sending automated queries. To protect our users, we can't process your request right now."
'''


# SECOP 1 - Chromedriver






























































# SECOP 2

# secop2_2020_UM_alimentos = secop2_2020_UM_alimentos.iloc[1965:]

problematic = []

for i in range(secop2_2020_UM_alimentos.shape[0]):

    id_contrato = secop2_2020_UM_alimentos['ID del Proceso'].iloc[i]
    
    path = r'D:\SECOP\2020\SECOP_II' + '\\' + id_contrato
    
    try:
        os.mkdir(path)
    except:
        print('Folder already created')
        pass
    
    profile = webdriver.FirefoxProfile()
    
    # Cookies
    profile.set_preference('browser.cache.disk.enable', False)
    profile.set_preference('browser.cache.memory.enable', False)
    profile.set_preference('browser.cache.offline.enable', False)
    profile.set_preference('network.cookie.cookieBehavior', 2)
    profile.set_preference("browser.privatebrowsing.autostart", True)
    profile.set_preference("security.fileuri.strict_origin_policy", False)
    profile.update_preferences()
    
    # Download options
    
    options = Options()
    options.set_preference("browser.download.folderList",2)
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference("browser.download.dir",path)
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf,application/vnd.adobe.xfdf,application/vnd.fdf,application/vnd.adobe.xdp+xml")
    profile.set_preference("pdfjs.disabled", True)
    
    
    # Capabilities
    capabilities = webdriver.DesiredCapabilities.FIREFOX
    capabilities['marionette'] = True
    
    
    firefox_binary = r'C:\bin\gecko\bin\geckodriver.exe'  
    driver = webdriver.Firefox(executable_path=firefox_binary, 
                               firefox_profile=profile,
                               capabilities = capabilities,
                               options = options)
    
    driver.delete_all_cookies()
    
    driver.get(secop2_2020_UM_alimentos['URLProceso'].iloc[i])
    
    # click en botón 'ver contrato'
    
    time.sleep(10)
    
    try:
    
        # Botones descargar    
        
        files = driver.find_elements_by_xpath('//*[@title="Descargar"]')

        for i in range(0,len(files)):
            time.sleep(3)
            if files[i].is_displayed():
                files[i].click()
    
        # WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.NAME, "Descargar"))).click()
    
        # WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.ID, "lnkSalesContractViewLink_0"))).click()
        
        # driver.find_element_by_id('lnkSalesContractViewLink_0').click()
        
        # Change frame 
        # all_frames = driver.find_elements_by_tag_name('iframe')
        # driver.switch_to.frame(all_frames[0])
        
        # time.sleep(10)
        
        # WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.ID, "lnkDownloadDocument_0"))).click()
        
        driver.close()
        
        print(id_contrato, "OK")
    
    except:
        driver.close()
        print(id_contrato, "problematic")
        problematic.append(id_contrato)
        pass



driver.close()
    
# Se descargan todos los documentos y se buscan los pdf donde haya información de alimentos y precios
# Siempre el último documento descargado es el contrato del proceso
# ~2000 contratos se demoraron ~30 horas



