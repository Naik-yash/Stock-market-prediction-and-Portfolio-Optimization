#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 22:45:54 2018

@author: yash
"""

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import sentiment_mod as s





#Keys Hidden 

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        
        tweet = all_data["text"]
        
        
       

        sentimentv, confidence = s.sentiment(tweet)
        

        print(tweet,sentimentv, confidence)
        if confidence*100 >= 70:
            output= open("/Users/yash/Downloads/twitter-out.txt","a")
            output.write(sentimentv)
            output.write("/n")
            output.close()
        
        return True

    def on_error(self, status):
        print (status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["dowjones"])





