from bs4 import BeautifulSoup as BS
import requests as req
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import itertools
from datetime import datetime

n = 150  # max_pages
news_urls = ['https://cryptoslate.com/news/bitcoin/page/' +
             str(i) for i in range(1, n)]
session = req.Session()


def scrapping_title(url):
    webpage = session.get(url)
    html_doc = webpage.text
    soup = BS(html_doc, "lxml")
    news_feed = soup.select_one('div.news-feed.slate')
    news_info = news_feed.select('a')
    news_title_data = []
    news_title_data = [{
        'title': news.get('title'),
        'url': news.get('href')
    } for news in news_info if news.get('title')]
    return news_title_data


def scrapping_info(url):
    webpage = session.get(url)
    html_doc = webpage.text
    soup = BS(html_doc, "lxml")
    author_info = soup.select_one('div.author-info')
    news_info = {
        'time': author_info.select_one('span.time').string[3:-4],
        'date': author_info.select_one('div.post-date').contents[0]
    }
    return news_info


with ThreadPoolExecutor() as executor:
    news_data = list(executor.map(scrapping_title, news_urls))
    news_data = list(itertools.chain(*news_data))
news_data = list(map(dict, set(map(frozenset, map(dict.items, news_data)))))
news_info_urls = list(map(lambda x: x['url'], news_data))

with ThreadPoolExecutor() as executor:
    news_infos = list(executor.map(scrapping_info, news_info_urls))

news_all = [{**d1, **d2} for d1, d2 in zip(news_data, news_infos)]
df = pd.DataFrame(news_all)
df.date = df.date.apply(lambda x: datetime.strptime(
    x.strip(), '%b. %d, %Y').date())
df.time = df.time.apply(
    lambda x: datetime.strptime(x.strip(), '%I:%M %p').time())
df.sort_values('date', ascending=False, inplace=True)
df.to_csv('C:/Users/jleung/workspace/test/news_data.csv',
          encoding="utf-8-sig", index=False)
