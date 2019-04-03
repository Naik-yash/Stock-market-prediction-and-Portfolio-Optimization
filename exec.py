#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:54:50 2018

@author: yash
"""

import sentiment_mod as s
import csv

docs=[]
with open("/Users/yash/Downloads/headlines.csv","r") as f:
    reader=csv.reader(f)
    for line in reader:
        print(s.sentiment(docs.append(line[2])))