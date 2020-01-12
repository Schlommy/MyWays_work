"""
    Author: Mudit Soni
    Used for scraping list of review pages for all the companies on AmbitionBox.
"""
import selenium
import pandas as pd
from selenium import webdriver
import time
from selenium.webdriver.firefox.options import Options
import pickle

opp = Options()
opp.add_argument('-headless')
driver= webdriver.Firefox()
industries=[]
companies= {}
website= "https://www.ambitionbox.com"

def initialize(driver):
	driver.get('https://www.ambitionbox.com/list-of-companies')
	driver.find_element_by_xpath('/html/body/div/div/div/div[2]/div/div[2]/div/div/aside/div[4]/p').click()

initialize(driver)

for i in range(1,65):
	ind= driver.find_element_by_xpath("//label[@for='IndustryName"+str(i)+"']")
	
	industries.append((ind.get_attribute("title"), ind.find_element_by_tag_name('span').text, 
							driver.find_element_by_id("IndustryName"+str(i)).get_attribute('value')))
#print(industries)

inds= {}
j= 0
start= 1
p_start= 1

flag= False
if flag==True:
	fil= ''
	with open(fil+' comp', 'rb') as f:
		companies= pickle.load(f)
	with open(fil+' ind', 'rb') as f:
		inds= pickle.load(f)
	start, p_start= (int(i) for i in fil.split())

for i in range(start, 65):
	#WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, "IndustryName"+str(i)))).click()
	link= website+"/list-of-companies?IndustryName="+industries[i-1][2]
	inds[industries[i-1][0]]=[]
	driver.get(link)
	num_comps= int(industries[i-1][1].replace('(','').replace(')','').replace(',',''))
	num_pages= num_comps//30 +1
	for page in range(p_start, num_pages+1):
		j+=1
		if j%200==0:
			driver.close()
			driver= webdriver.Firefox()
		driver.get(link+"&page="+str(page))
		time.sleep(0.4)
		print(i, str(page)+'/'+str(num_pages), end='\r')
		try:
			company_list= driver.find_elements_by_class_name("company-result-card-wrapper")
		except:
			continue
		for company in company_list:
			try:
				name= company.find_element_by_xpath(".//a[1]/h2").get_attribute('title')
			except selenium.common.exceptions.NoSuchElementException:
				name=''
			#print(name)
			if name in companies.keys():
				companies[name]['industry'].append(industries[i-1][0])
				continue
			else:
				companies[name]= {}
				companies[name]['industry']= []
				companies[name]['industry'].append(industries[i-1][0])
				l= company.find_element_by_xpath(".//a[1]").get_attribute('href')
				companies[name]['link']= website+"/reviews/"+l[37:-9]+"-reviews"
			inds[industries[i-1][0]].append(name)
		
		if j%400==0:
			with open(str(i)+' '+str(page+1)+' comp', 'wb') as f:
				pickle.dump(companies, f)
			with open(str(i)+' '+str(page+1)+' ind', 'wb') as f:
				pickle.dump(inds, f)
			j=0

print(inds)
with open("industries.pkl","wb") as f:
	pickle.dump(inds, f)
with open("companies.pkl", "wb") as f:
	pickle.dump(companies, f)

