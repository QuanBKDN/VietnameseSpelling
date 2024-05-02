import os
import urllib

from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
DATABASE_ADDRESS = os.getenv("DATABASE_ADDRESS")
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

CONNECTION_STRING = (
    "mongodb://"
    + urllib.parse.quote(USERNAME)
    + ":"
    + urllib.parse.quote(PASSWORD)
    + DATABASE_ADDRESS
)


def get_connection(collection_name=COLLECTION_NAME):
    client = MongoClient(CONNECTION_STRING)
    if DATABASE_NAME in client.list_database_names():
        print("Database exist")
    else:
        print("Database does not exist, created newone")
    connection = client[DATABASE_NAME]
    if collection_name in connection.list_collection_names():
        print("Collection exist")
    else:
        print(f" Collection{collection_name} does not exist, created newone")
    return connection[collection_name]


def insert_log(collection, log_item):
    collection.insert_one(log_item.dict())
