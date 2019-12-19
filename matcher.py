"""
	Author: Mudit Soni
	Script to get indices of companies from roles.csv (Scraped data) in Companies_final_2018.csv (Government data)
	by matching company names with fuzzy matching.
"""
import pandas as pd
import numpy as np
from hr_model import *
from CompanyMatcher import *
from preprocess_company import *
from fuzzywuzzy import fuzz
import pickle

gov_data= pd.read_csv('Companies_Final_2018.csv')
data= pd.read_csv('roles (1).csv')

gov_names= gov_data['COMPANY_NAME'].apply(get_company)
names= data['Company Name'].apply(get_company)
final_lst_to_search_on = [fuzz.SequenceMatcher(isjunk=None, 
                            seq1=fuzz._process_and_sort(el, force_ascii=True, full_process=True)) for el in gov_names]

final_matches = []
start= 0

#For loading checkpoint file
flag=False
if flag==True:
    fil='_'
    with open(fil,'rb') as f:
        final_matches= pickle.load(f)
    start= int(fil[:-1])

for i in range(start, len(names.unique())):
    x= names.unique()[i]
    print(i, ' / ', len(names.unique()), end='\r')
    best_score = 0
    best_match = None
    for j, matcher in enumerate(final_lst_to_search_on):
        matcher.set_seq2(fuzz._process_and_sort(x, force_ascii=True, full_process=True))
        ratio = matcher.quick_ratio()
        if ratio > best_score:
            best_score= ratio
            best_match= j
    final_matches.append((x, best_match, final_lst_to_search_on[best_match], best_score))
    if i%500==0 and i!=0:
        with open(str(i+1)+' final_matches.pkl','wb') as f:
            pickle.dump(final_matches, f)

#print(final_matches)
with open('final_matches.pkl','wb') as f:
    pickle.dump(final_matches, f)
