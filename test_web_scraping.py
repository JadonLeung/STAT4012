from bs4 import BeautifulSoup as BS
import requests as req
import pandas as pd

n = 10 # max_pages
news_titles = []
news_url = []

for i in range(1, n):
    url = 'https://cryptoslate.com/news/bitcoin/page/' + str(i)
    webpage = req.get(url)
    html_doc = webpage.text
    soup = BS(html_doc, "lxml")
    news_feed = soup.find_all('div', class_ = "news-feed slate")
    news_info = news_feed[0].find_all('a')
    for news in news_info:
        if news.get('title') == None:
            continue
        news_titles.append(news.get('title'))
        news_url.append(news.get('href'))

news_titles = pd.unique(news_titles)
news_url = pd.unique(news_url)

df = pd.DataFrame({'titles':news_titles, 'url':news_url})
df.to_csv('news_data.csv')





