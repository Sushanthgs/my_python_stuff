# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 11:25:11 2021

@author: sushanthsgradlaptop2
"""

import requests
from bs4 import BeautifulSoup
start_url='https://quotes.toscrape.com'
from time import sleep
from random import choice
def get_quotes_on_page(res_text):
    quote_list_raw=res_text.find_all(class_='quote')
    q_text_fil=[]
    for q in quote_list_raw:
            q_text_fil.append({"text":q.find(class_='text').get_text(),
                   "author":q.find(class_='author').get_text(),
                   "bio-link":q.find('a')['href']
        })
    return(q_text_fil)
    
def get_new_page_link(res_text):
    next_link=res_text.find(class_='next')
    new_page_link=next_link.find('a')['href'] if next_link else None
    return(new_page_link)



def scrape_pages(*args):
    if(len(args)>1):
        numpages=args[1]
        for i in range(1,numpages+1):
            if i==1:
                url_scrape=start_url+'/page/1'
                res=requests.get(url_scrape)
                res_text=BeautifulSoup(res.text,'html.parser')
                q_text_all=get_quotes_on_page(res_text)
            else:
                npg=get_new_page_link(res_text)
                if(not npg):
                    break
                url_scrape=start_url+npg
                res=requests.get(url_scrape)
                res_text=BeautifulSoup(res.text,'html.parser')
                q_text_all.extend(get_quotes_on_page(res_text))
                sleep(2)
    else:
        k=1
        url_scrape=start_url+'/page/1'
        res=requests.get(url_scrape)
        res_text=BeautifulSoup(res.text,'html.parser')
        q_text_all=get_quotes_on_page(res_text)
        npg=1
        while npg:
            npg=get_new_page_link(res_text)
            if(not npg):
                break
            url_scrape=start_url+npg
            res=requests.get(url_scrape)
            res_text=BeautifulSoup(res.text,'html.parser')
            q_text_all.extend(get_quotes_on_page(res_text))
            sleep(2)
    return(q_text_all)

#%%
def get_hint(num_guesses,quote_choice):
    if(num_guesses==3):
        res=requests.get(start_url+'/'+quote_choice['bio-link'])
        res_text=BeautifulSoup(res.text,'html.parser')
        born_loc=res_text.find(class_='author-born-location').get_text()
        born_date=res_text.find(class_='author-born-date').get_text()
        print(f"I was born in {born_loc} on {born_date}")
    elif(num_guesses==2):
        print(f"My first name starts with {quote_choice['author'][0]}")
    elif(num_guesses==1):
        l_name=quote_choice['author'].split(' ')
        print(f"My last name starts with {l_name[1][0]}")
    else:
        print(f"You're out of guesses, I'm {quote_choice['author']}")
        
def get_pts(num_guesses):
    if(num_guesses==3):
        pts=5
    elif(num_guesses==2):
        pts=3
    elif(num_guesses==1):
        pts=1
    else:
        pts=0
    return(pts)
    
#%%
  
    
q_pgs_all=scrape_pages(start_url)
  
#%%
total_pts=0
c='y'
num_guesses=3
while (c!='n' and num_guesses>0):
    if(num_guesses==3):
        quote_choice=choice(q_pgs_all)
        print(quote_choice['text'])
        
    guess_author=input('Who said this?')
    if(guess_author.lower()==quote_choice['author'].lower() and num_guesses>0):
        print('Correct')
        pts=get_pts(num_guesses)
        total_pts+=pts
        print(f'Your score is :{(pts)}')
        print(f'Your score overall is:{(total_pts)}')
        num_guesses=0
    else:
        print('Wrong')
        num_guesses-=1
        print('Here is a hint')
        get_hint(num_guesses,quote_choice)
    
    if(c=='n'):
        break
   
 
    if(num_guesses==0):
        c=input('Play again? y/n')
        if(c=='y'):
            num_guesses=3
    
    
    

