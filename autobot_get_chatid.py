import requests

TOKEN = "8266827966:AAH-giw0iEsUBxMtSBss8qJrqa_WF4bGs5s"
url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

response = requests.get(url).json()
print(response)