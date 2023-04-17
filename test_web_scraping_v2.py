from bs4 import BeautifulSoup as BS
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import itertools
from datetime import datetime
import re


def scrapping(file_name):
    with open(file_name, "r") as HTMLFile:
        # Reading the file
        index = HTMLFile.read()
    S = BS(index, 'lxml')
    heading_tags = S.find_all('b', class_='printheadline enHeadline')
    fields_tags = S.find_all('div', class_='leadFields')
    news_data = [{
        'heading': re.sub(r' -- (Barrons.com|WSJ)', '', tag1.text),
        'news_provider': tag2.text.split(',')[0],
        'date': tag2.text.split(',')[2],
        'time': tag2.text.split(',')[1],
        'wordcount': tag2.text.split(',')[3]
    } for tag1, tag2 in zip(heading_tags, fields_tags) if len(tag2.text.split(',')) == 5]
    return news_data

# n = 30 # number of files (Web news)
n = 26 # number of files (Dow Jones)
# file_list = ['Factiva/Factiva_{}.html'.format(i) for i in range(1, n + 1)]
file_list = ['Dow_Jones/Factiva_{}.html'.format(i) for i in range(1, n + 1)]

with ThreadPoolExecutor() as executor:
    news_data = list(executor.map(scrapping, file_list))
    news_data = list(itertools.chain(*news_data))

df = pd.DataFrame(news_data)
df.date = df.date.apply(lambda x: datetime.strptime(
    x.strip(), '%d %B %Y').date())
df.time = df.time.apply(
    lambda x: datetime.strptime(x.strip(), '%H:%M %Z').time())
df.wordcount = df.wordcount.apply(lambda x: int(x[:-5]))
df.sort_values(['date', 'time'], ascending=False, inplace=True)
df.drop_duplicates('heading', keep='last', inplace=True)
df.to_csv('C:/Users/jleung/workspace/test/dow_jones_news_data.csv',
          encoding="utf-8-sig", index=False)
