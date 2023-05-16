import datetime

from src.scrappers.gettyimages_scrapper import GettyImagesScrapper
from src.scrappers.deia_scrapper import DeiaScrapper

NUM_PAGES = 1
QUERIES = ['pais vasco']

QUERIES2 = ['Bilbao']
START_DATE = (2021, 1, 1)
END_DATE = (2023, 5, 1)

if __name__ == '__main__':
    #gettyimage_scrapper = GettyImagesScrapper(NUM_PAGES, QUERIES)
    #gettyimage_scrapper.scrap()

    deia_scrapper = DeiaScrapper(START_DATE, END_DATE, QUERIES2)
    deia_scrapper.scrap()
    