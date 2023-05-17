import datetime

from src.scrappers.gettyimages_scrapper import GettyImagesScrapper
from src.scrappers.deia_scrapper import DeiaScrapper

# General config
QUERIES = ['bilbao']
# GettyImages Config
NUM_PAGES = 1
# Deia Config
START_DATE = datetime.datetime(2022, 5, 17)
END_DATE = datetime.datetime(2023, 5, 17)

if __name__ == '__main__':
    #gettyimage_scrapper = GettyImagesScrapper(NUM_PAGES, QUERIES)
    #gettyimage_scrapper.scrap()

    deia_scrapper = DeiaScrapper(START_DATE, END_DATE, QUERIES)
    deia_scrapper.scrap()
    