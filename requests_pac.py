from dotenv import load_dotenv
import os
import json
import requests

r = requests.get('https://api.github.com/events',stream=True)
# rc = r.content
# for text in rc:
    # print(text,flush=True)
# print(r.url)
# r.json()
# print(r.text)
# print(r.content)

r.raw
# <urllib3.response.HTTPResponse object at 0x101194810>

r.raw.read(10)
