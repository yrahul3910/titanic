# -*- coding: utf-8 -*-
from dotenv import load_dotenv, find_dotenv
import os
import requests
from requests import session
import logging

payload = {
    "action": "login",
    "username": os.environ.get("KAGGLE_USERNAME"),
    "password": os.environ.get("KAGGLE_PASSWORD")
}

def extract_data(url, file_path):
    """
    Extract data from Kaggle
    """
    with session() as c:
        c.post("https://www.kaggle.com/account/login", data=payload)
        with open(file_path, "wb") as f:
            response = c.get(url, stream=True)
            for block in response.iter_content(1024):
                f.write(block)


def main(project_dir):
    logger = logging.getLogger(__name__)
    logger.info("getting raw data")

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    train_url = "https://www.kaggle.com/c/titanic/download/train.csv"
    test_url = "https://www.kaggle.com/c/titanic/download/test.csv"

    raw_data_path = os.path.join(project_dir, "data", "raw")
    train_data_path = os.path.join(raw_data_path, "train.csv")
    test_data_path = os.path.join(raw_data_path, "test.csv")

    extract_data(train_url, train_data_path)
    extract_data(test_url, test_data_path)
    logger.info("done!")


if __name__ == "__main__":
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    
    main(project_dir)