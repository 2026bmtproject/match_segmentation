import os

from google import genai

'''
how to get api key:
打開 aistudio.google.com/app/apikey，創建一個新的 API 金鑰，然後將金鑰值複製到環境變數 GEMINI_API_KEY 中。
$env:GEMINI_API_KEY = "你的_api_key"
'''

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise SystemExit(
        "找不到 Gemini API key。請先在目前 PowerShell 視窗設定 $env:GEMINI_API_KEY = \"你的_api_key\"，"
        "或 $env:GOOGLE_API_KEY = \"你的_api_key\"，再重新執行。"
    )

client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello, test!"
)
print(response.text)