import random

PROXIES_FILE_PATH = "proxies.txt"

class ScrappingUtils():

    def __init__(self) -> None:
        with open(PROXIES_FILE_PATH, 'r') as f:
            self.proxy_list = f.read().splitlines()


    def get_random_proxy(self):
        proxy = random.choice(self.proxy_list)
        proxies = {
            "http": 'http://' + proxy,
        }
        return proxies