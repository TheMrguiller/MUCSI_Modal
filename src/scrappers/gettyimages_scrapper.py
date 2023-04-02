import requests
import numpy as np
import csv
import random
import os

from datetime import datetime
from fake_useragent import UserAgent
from tqdm import tqdm
from bs4 import BeautifulSoup

from .scrapping_utils import ScrappingUtils

class GettyImagesScrapper(ScrappingUtils):

    def __init__(self, num_pages, queries) -> None:
        super().__init__()
        self.num_pages = num_pages
        self.queries = queries

    def query_request(self, base_url, i):
        ua = UserAgent()
        headers = {'User-Agent':ua.random}
        succes_response = False
        while not succes_response:
            proxies = self.get_random_proxy()
            print("Trying with proxies: {}".format(proxies))
            try:
                response = requests.get(base_url + '?page={0}'.format(i), headers=headers, proxies=proxies)
                if response.status_code == 200:
                    succes_response = True
            except:
                random.randint(2, 10)
        return response
    
    def image_detail_request(self, detail_url):
        ua = UserAgent()
        headers = {'User-Agent':ua.random}
        succes_response = False
        while not succes_response:
            proxies = self.get_random_proxy()
            print("Trying with proxies: {}".format(proxies))
            try:
                response = requests.get(detail_url, headers=headers, proxies=proxies)
                if response.status_code == 200:
                    succes_response = True
            except:
                random.randint(2, 10)
        return response
    
    def parse_image_detail_html_data(self, response):
        soup = BeautifulSoup(response.content, "html.parser")
        caption_div_elem = soup.find("div", {"data-testid":"caption"})

        if caption_div_elem:
            return caption_div_elem.contents
    
    def download_image(self, url):
        path = "data/imgs/"
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
    
    def parse_query_html_data(self, response):
        results = []
        soup = BeautifulSoup(response.content, "html.parser")

        img_divs = soup.find_all("div", {"data-testid": "galleryMosaicAsset"})

        for div in img_divs:
            img_url = None
            local_path = None
            title = None
            caption = None

            # URL
            img_elem = div.find("img", {"class": "yvFP5yDnHgdkSkBpoY3V"})
            if img_elem:
                img_url = img_elem["src"]
                local_path = self.download_image(img_url)
                title = img_elem["alt"]

                detail_url_elem = div.find("a", {"class": "wwW2JD5Y0CMfeJ8BD1xP"})
                if detail_url_elem:
                    # Caption
                    detail_url = 'https://www.gettyimages.es' + detail_url_elem['href']
                    print(detail_url)
                    response = self.image_detail_request(detail_url)
                    caption = self.parse_image_detail_html_data(response)

                    results.append([img_url, local_path, title, caption])

        return results
    
    def result_to_csv(self, results_list, path, query):
        if not os.path.exists(path): 
            os.mkdir(path)
        filename = "gettyimages_" + query.replace(" ", "_") + ".csv"
        with open(path+filename, "w", encoding="utf-8", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["img_url", "local_path", "title", "caption"])
            writer.writerows(results_list)

    def scrap(self):
        for query in self.queries:
            print("Starting query: {}".format(query))
            search_query = query.replace(' ', '-')
            gettyimagge_base_url = 'https://www.gettyimages.es/fotos/{0}'.format(search_query)
            
            final_results_list = []
            for i in tqdm(range(1, self.num_pages+1)):
                response = self.query_request(gettyimagge_base_url, i)
                results = self.parse_query_html_data_(response)

                if len(final_results_list) == 0:
                    final_results_list = results
                else:
                    final_results_list = np.concatenate((results, final_results_list), axis=0)
            
            print("{} images scrapped".format(len(final_results_list)))

            if len(final_results_list) > 0:
                path = "./data/captions/raw/"
                self.result_to_csv(final_results_list, path, query)
