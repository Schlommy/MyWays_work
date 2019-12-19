"""
	Author: Mudit Soni
	Used for scraping individual reviews from list of company pages obtained from scraper.py.
"""
import pandas as pd
from bs4 import BeautifulSoup
import pickle
#import shelve
import requests
import numpy as np
import sys
#for allowing pickling of large files:
sys.setrecursionlimit(60000) 

def extract_source(url):
	"""
		Used to bypass permission requirements.
	"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    source=requests.get(url, headers=headers).text
    return source

with open("companies.pkl", "rb") as f:
	companies= pickle.load(f)
with open("industries.pkl", "rb") as f:
	inds= pickle.load(f)

start= 0 
n_start= 0
sample_comps= {}

#flag is used to load in a checkpoint file
flag= False
if flag==True:
	fil= ''
	with open(fil, 'rb') as f:
		#companies= pickle.load(f)
		sample_comps= pickle.load(f)
	#with shelve.open(fil, 'r') as shelf:
	#	sample_comps= shelf
	x_start, k_start, i_start, j_start= (int(i) for i in fil.split()[:4])

def get_all_companies(companies, start= 0, n_start= 0):
	"""
		Get company-wise review data. Checks for each and every company irrespective of industry.
	"""
	df= pd.DataFrame(columns=['Job', 'Division', 'City', 'Duration', 'Date', 'Overall Rating', 'Skill development/learning', 
								'Work-Life balance', 'Compensation & Benefits', 'Company culture', 'Job Security', 
								'Career growth & opportunities', 'Work Satisfaction', 'Likes', 'Dislikes', 'Work', 'Working Days', 
								'Work Timings', 'Job Travel', 'Facilities'])
	n=0
	for k in range(start, len(companies.keys())):
		name= companies.keys()[k]
		df = df.iloc[0:0]
		if flag==True:
			df= companies[name]['Reviews']
		url= companies[name]['link']
		page = requests.get(url)
		soup = BeautifulSoup(extract_source(url), 'html5lib')
		
		companies[name]['Overall Rating']= soup.find('span', class_="rating_text").get_text()[:3]
		avg_ratings= card.find_all('div', class_='rating_display_btn')
		for avg_rating in avg_ratings:
			companies[name][avg_rating.find('p').get_text()]= avg_rating.find('span').get_text()[:3]

		temp= soup.find('span', class_='card-meta')
		temp= temp.find_all('span')[0].get_text()
		num_reviews= int(temp[temp.find('f')+2: temp.find('R')-1])
		num_pages= num_reviews//10+1
	
		j=0
		for i in range(num_pages):
			n+=1
			print(k, name, str(i)+' / '+str(num_pages), end='\r')
			if i==0:
				pass
			else:
				link= url+"?page="+str(i+1)+"&"	
				page = requests.get(link)
				soup = BeautifulSoup(extract_source(link), 'html5lib')
		
			review_cards= soup.find_all('div', class_='review_card')
			for card in review_cards:
				try:
					txt= card.find('h2', itemprop='name').get_text()
				#print(txt[:txt.find('for')].strip())
					df.loc[j, 'Job']= txt[:txt.find(' for ')].strip()
					df.loc[j, 'Duration']= txt[txt.find(' for ')+5:txt.find(' in ')].strip()	
					df.loc[j, 'City']= txt[txt.find(' in ')+4:].strip()
				except:
					pass
				
				try:
					txt= card.find('span', class_='misc_text_mobile').get_text()
					df.loc[j, 'Division']= txt[txt.find('|')+2:]
				except:
					pass
			
				#print(card.find('div', class_='rate-wrap')['title'])
				df.loc[j, 'Overall Rating']= card.find('div', class_='rate-wrap')['title'][-3]			
		
				ratings= card.find_all('div', class_='detailed_ratings')
				for rating in ratings:
					df.loc[j, rating.find('label').get_text()]= rating.find('div')['title'][-1]
			
				texts= card.find_all('div', class_='review_text_wrap')
				for text in texts:
					try:
						df.loc[j, text.find('h3').get_text()]= text.find('p').get_text()
					except:
						pass
	
				texts= card.find_all('div', class_='job_description_tile')
				for text in texts:
					df.loc[j, text.find('h3').get_text()]= text.find('p').get_text().strip()
		
				facilities= card.find_all('p', class_='skill_label')
				df.loc[j, 'Facilities']=''
				for facility in facilities:
					df.loc[j, 'Facilities']+= facility.get_text().strip()+','
		
				df.loc[j, 'Date']= card.find('time')['datetime']
			
				j+=1
				#break
			
			if (n+1)%5000==0:
				companies[name]['Reviews']= df
				with open(str(k)+' '+str(i+1)+' _', 'wb') as f:
					pickle.dump(companies, f)
				n=0
			#break
		companies[name]['Reviews']= df

	df.to_csv('test.csv')
	with open('companies2.pkl', 'wb') as f:
		pickle.dump(companies, f)
		 
def sample(inds, companies, sample_comps, flag, x_start=0, k_start=0, i_start=0, j_start=0):
	"""
		Get industry wise review data. Allows for spacifying number of companies for each industry 
		and number of review pages for each company.
	"""
	df= pd.DataFrame(columns=['Job', 'Division', 'City', 'Duration', 'Date', 'Overall Rating', 'Skill development/learning', 
								'Work-Life balance', 'Compensation & Benefits', 'Company culture', 'Job Security', 
								'Career growth & opportunities', 'Work Satisfaction', 'Likes', 'Dislikes', 'Work', 'Working Days',
								'Work Timings', 'Job Travel', 'Facilities'])
	n=0
	#sample_comps= {}
	#t=0
	for x in range(x_start, len(inds.keys())):
		ind= list(inds.keys())[x]
		#t+=1

		num_comps= 50 if len(inds[ind])>2 else len(inds[ind])

		for k in range(k_start, num_comps):
			name= inds[ind][k]
			#print(ind, k)
			sample_comps[name]= companies[name]
			df = df.iloc[0:0]
			j= 0
			if flag==True:
				df= sample_comps[name]['Reviews']
				j= j_start
				flag= False		
	
			url= companies[name]['link']
			page = requests.get(url)
			soup = BeautifulSoup(extract_source(url), 'html5lib')

			sample_comps[name]['Overall Rating']= soup.find('span', class_="rating_text").get_text()[:3]
			avg_ratings= soup.find_all('div', class_='rating_display_btn')
			for avg_rating in avg_ratings:
				sample_comps[name][avg_rating.find('p').get_text()]= avg_rating.find('span').get_text()[:3]

			temp= soup.find('span', class_='card-meta')
			temp= temp.find_all('span')[0].get_text()
			num_reviews= int(temp[temp.find('f')+2: temp.find('R')-1])
			num_pages= num_reviews//10+1
			num_pages= 10 if num_pages>10 else num_pages			

			for i in range(i_start, num_pages):
				n+=1
				print(ind, str(k)+' / '+str(num_comps), str(i)+' / '+str(num_pages), end='\r')
				if i==0:
					pass
				else:
					link= url+"?page="+str(i+1)+"&"	
					page = requests.get(link)
					soup = BeautifulSoup(extract_source(link), 'html5lib')
			
				review_cards= soup.find_all('div', class_='review_card')
				for card in review_cards:
					try:
						txt= card.find('h2', itemprop='name').get_text()
						df.loc[j, 'Job']= txt[:txt.find(' for ')].strip()
						df.loc[j, 'Duration']= txt[txt.find(' for ')+5:txt.find(' in ')].strip()	
						df.loc[j, 'City']= txt[txt.find(' in ')+4:].strip()
					except:
						pass
				
					try:
						txt= card.find('span', class_='misc_text_mobile').get_text()
						df.loc[j, 'Division']= txt[txt.find('|')+2:]
					except:
						pass
			
					df.loc[j, 'Overall Rating']= card.find('div', class_='rate-wrap')['title'][-3]			
		
					ratings= card.find_all('div', class_='detailed_ratings')
					for rating in ratings:
						df.loc[j, rating.find('label').get_text()]= rating.find('div')['title'][-1]
			
					texts= card.find_all('div', class_='review_text_wrap')
					for text in texts:
						try:
							df.loc[j, text.find('h3').get_text()]= text.find('p').get_text()
						except:
							pass

					texts= card.find_all('div', class_='job_description_tile')
					for text in texts:
						df.loc[j, text.find('h3').get_text()]= text.find('p').get_text().strip()
			
					facilities= card.find_all('p', class_='skill_label')
					df.loc[j, 'Facilities']=''
					for facility in facilities:
						df.loc[j, 'Facilities']+= facility.get_text().strip()+','
			
					df.loc[j, 'Date']= card.find('time')['datetime']
				
					j+=1

				if (n+1)%2500==0:
					sample_comps[name]['Reviews']= df
					with open(str(x)+' '+str(k)+' '+str(i)+' '+str(j-1)+' _', 'wb') as f:
						pickle.dump(sample_comps, f)
					#with shelve.open(str(x)+' '+str(k)+' '+str(i)+' '+str(j-1)+' _') as shelf2:
    					#shelf2=sample_comps
					n=0

			sample_comps[name]['Reviews']= df
			i_start=0
		k_start=0
			
		#if t==1:
		#	break
		#df.to_csv('samps.csv')
	with open('sample_companies.pkl', 'wb') as f:
		pickle.dump(sample_comps, f)
	#with shelve.open('sample_companies2', 'c') as shelf:
    	#	shelf=sample_comps
	

sample(inds, companies, sample_comps, flag)
#print(inds.keys())
#print(companies['Bentel'])
#print(len(companies.keys()))
			
