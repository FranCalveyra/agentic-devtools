import os

from dotenv import load_dotenv


class Environment:
    def __init__(self):
        load_dotenv()
        self.GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
