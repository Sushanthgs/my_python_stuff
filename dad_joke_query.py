# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:37:57 2020

@author: sushanthsgradlaptop2
"""

import requests 
from random import choice

topic=input('what would you like to search for?')
url='https://icanhazdadjoke.com/search'
res=requests.get(url,
                      headers={'Accept':'application/json'},
                      params={'term':topic})
json_res=res.json()
results=json_res['results']
if(json_res['total_jokes']==0):
    print(f'Sorry, dont have a joke about {topic} yet')
elif(json_res['total_jokes']==1):
    print(f'Here\'s the only joke I have about{topic}' )
    print(json_res['results'][0])
else:
    print('I have many jokes about that')
    ch=input('Do you want all of them (y/n) ?')
    if(ch=='n'):
        print('ok, here is one at random')
        print(choice(results)['joke'])
    elif(ch=='y'):
        for k in results:
            print(k['joke'])
    else:
        print('You typed something wrong, try again')
       
        
    