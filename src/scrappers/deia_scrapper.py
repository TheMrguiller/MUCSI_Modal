import pandas as pd
import requests
import os
import csv
import random
import numpy as np

from tqdm import tqdm
from urllib.request import urlopen
from bs4 import BeautifulSoup as BS
from urllib.parse import urlparse, urlsplit
from PIL import Image
from datetime import datetime, timedelta, date
from fake_useragent import UserAgent

from .scrapping_utils import ScrappingUtils

class DeiaScrapper(ScrappingUtils):

    def __init__(self, start_date, end_date, queries) -> None:
        super().__init__()
        self.start_date = start_date
        self.end_date = end_date
        self.delta = timedelta(days=1)
        self.current_date = start_date
        self.queries = queries
    
    def url_request(self, url):
        ua = UserAgent()
        headers = {'User-Agent':ua.random}
        succes_response = False
        while not succes_response:
            proxies = self.get_random_proxy()
            #print("Trying with proxies: {}".format(proxies))
            try:
                response = requests.get(url, headers=headers, proxies=proxies)
                if response.status_code == 200:
                    succes_response = True
            except:
                random.randint(2, 10)
        return response
    
    def parse_query_html_data(self, response):
        results = []
        html = response.content
        soup = BS(html, "html.parser")
        section = soup.find('section',"news-module-subsection news-module-subsection--globals")
        articles = section.find_all('article')
        for element in articles:

            imagen = element.find('noscript')
            img = imagen.find('img')
            if img:
                a = element.find('a')
                href = a.get('href')
                
                news_res = self.url_request(href)
                news_html = news_res.content
                news_soup = BS(news_html, "html.parser")
                
                article = news_soup.find('article', "article-photo article-photo--full") 
                if article:
                    p_element = article.find('p')
                    img_big_src = article.find('picture').find('img')['src']
                    caption_text = str(p_element.find('em').text.replace("\"", " "))

                    try:
                        local_path = self.download_image(img_big_src)
                        results.append([img_big_src, local_path, caption_text])
                    except:
                        pass
                        
        return results

    def download_image(self, url):
        path = "data/imgs/deia/"
        ppid = os.getpid()
        timestamp = datetime.now().timestamp()
        response = requests.get(url)
        if response.status_code == 200:
            if not os.path.exists(path): os.mkdir(path)
            filename = "{}-{}.jpg".format(timestamp, ppid)
            with open(path + filename , 'wb') as f:
                f.write(response.content)
            return path+filename
        return None

    def result_to_csv(self, results_list, path, query):
        if not os.path.exists(path): 
            os.mkdir(path)
        filename = "deia_" + query.replace(" ", "_") + ".csv"
        with open(path+filename, "w", encoding="utf-8", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["img_url", "local_path", "caption"])
            writer.writerows(results_list)

    def scrap(self):
       for query in self.queries:
            print("Starting query: {}".format(query))
            #search_query = query.replace(' ', '-')
            current_date = self.start_date
            final_results_list = []
            for _ in tqdm(range((self.end_date - self.start_date).days + 1)):
                deia_base_url = "https://www.deia.eus/hemeroteca/{}/buscador/?text={}&type=article"
                res = self.url_request(deia_base_url.format(current_date.strftime("%Y/%m/%d"), query))
                results =  self.parse_query_html_data(res)
                
                if len(final_results_list) == 0:
                    final_results_list = results
                else:
                    final_results_list= np.concatenate((results, final_results_list), axis=0)
                current_date += self.delta
            
            print("{} images scrapped".format(len(final_results_list)))
            if len(final_results_list) > 0:
                path = "./data/captions/raw/"
                self.result_to_csv(final_results_list, path, query)
