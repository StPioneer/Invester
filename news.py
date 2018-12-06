import requests
import re
from bs4 import BeautifulSoup
base_url = "https://tw.news.yahoo.com"
url = "https://tw.news.yahoo.com/finance/archive/"
pages = range(1, 40)  # scrape for 40 pages, for each page with 25 news.

links = []
for page in pages:
    print(url + str(page) + ".html")
    print("===")
    yahoo_r = requests.get(url + str(page) + ".html")
    yahoo_soup = BeautifulSoup(yahoo_r.text, 'html.parser')
    finance = yahoo_soup.findAll('div', {'class': 'story'})
    for info in finance:
        link = ""
        try:
            link = info.findAll('a', href=True)[0]
            if link.get('href') != '#':
                links.append(base_url + link.get("href"))
                print(base_url + link.get("href"))
                print('===')
        except:
            link = None