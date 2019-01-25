from dotenv import find_dotenv, load_dotenv
import sys
import os

def getenv(name):
    val = os.getenv(name)
    return val

def load_env():
    load_dotenv(find_dotenv())
    return