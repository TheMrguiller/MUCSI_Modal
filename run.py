from src.scrappers.gettyimages_scrapper import GettyImagesScrapper


NUM_PAGES = 1
QUERIES = ['pais vasco']

if __name__ == '__main__':
    gettyimage_scrapper = GettyImagesScrapper(NUM_PAGES, QUERIES)
    gettyimage_scrapper.scrap()