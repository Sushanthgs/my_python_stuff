# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:45:06 2020

@author: sushanthsgradlaptop2
"""
from random import shuffle
class Card:
    def __init__(self,value,suit):
        self.value=value
        self.suit=suit
    def __repr__(self):
        return(f"{self.value} of {self.suit}")
C=Card('A','Clubs')
#%%
class Deck:
    suits=['Hearts','Clubs','Diamonds','Spades']
    values=['A','2','3','4','5','6','7','8','9','10','J','Q','K']
    
    def __init__(self):
        self.cards=[Card(v,s) for v in Deck.values for s in Deck.suits]
    def _deal(self,num):
        count=self.count()
        if(count==0):
            raise ValueError('All cards have been dealt')
        else:
            d_c=min(count,num)
            deal_cards=self.cards[-d_c:]
            self.cards=self.cards[:-d_c]
            
            return(deal_cards)
    def count(self):
        return (len(self.cards))
    def deal_card(self):
        self._deal(1)[0]
    def deal_hand(self,num):
        self._deal(num)
    def __repr__(self):
       return(f"Deck of {Deck.numCards} cards")
    def shuffle(self):
        if(len(self.cards)<52):
            raise ValueError('Only full decks can be shuffled')
        else:
            return(shuffle(self.cards))
        
        
d = Deck()
d.shuffle()
card = d.deal_card()
print(card)
hand = d.deal_hand(50)
card2 = d.deal_card()
print(card2)
print(d.cards)
card2 = d.deal_card()
