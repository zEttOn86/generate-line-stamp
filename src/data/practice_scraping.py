# coding:utf-8
import os, sys, time
import requests
from bs4 import BeautifulSoup

r = requests.get('https://news.yahoo.co.jp')

soup = BeautifulSoup(r.content, 'html.parser')
for i in soup.select("p.ttl"):
    print(i.getText())
