import requests
from bs4 import BeautifulSoup
import os
import time
import fitz
import re
import chardet

BASE_URL = "https://mis.ccu.edu.tw"

PAGES = {
    "首頁": "/?Lang=zh-tw",
    "教師介紹": "/p/412-1120-2478.php?Lang=zh-tw",
    "系主任": "/p/412-1120-2075.php?Lang=zh-tw",
    "兼任教師": "/p/412-1120-2477.php?Lang=zh-tw",
    "行政人員": "/p/412-1120-2079.php?Lang=zh-tw",
    "學士班資訊": "/p/412-1120-3300.php?Lang=zh-tw",
    "資管所資訊": "/p/412-1120-3301.php?Lang=zh-tw",
    "醫資所資訊": "/p/412-1120-3302.php?Lang=zh-tw",
    "碩專班資訊": "/p/412-1120-3303.php?Lang=zh-tw",
    "博士班資訊": "/p/412-1120-3304.php?Lang=zh-tw",
    "招生學士班": "/p/412-1120-2217.php?Lang=zh-tw",
    "招生碩士班": "/p/412-1120-2222.php?Lang=zh-tw",
    "招生碩專班": "/p/412-1120-2218.php?Lang=zh-tw",
    "招生博士班": "/p/412-1120-2219.php?Lang=zh-tw",
    "表單下載學士": "/p/412-1120-3333.php?Lang=zh-tw",
    "表單下載資管所": "/p/412-1120-3334.php?Lang=zh-tw",
    "表單下載醫資所": "/p/412-1120-3335.php?Lang=zh-tw",
    "表單下載碩專班": "/p/412-1120-3336.php?Lang=zh-tw",
    "表單下載博士班": "/p/412-1120-3337.php?Lang=zh-tw",
    "資管所相關規範": "/p/412-1120-3326.php?Lang=zh-tw",
    "資管所獎助學金": "/p/412-1120-3306.php?Lang=zh-tw",
    "醫資所相關規範": "/p/412-1120-3325.php?Lang=zh-tw",
    "醫資所獎助學金": "/p/412-1120-3329.php?Lang=zh-tw",
    "博士班相關規範": "/p/412-1120-3324.php?Lang=zh-tw",
    "博士班獎助學金": "/p/412-1120-3331.php?Lang=zh-tw",
    "系所介紹歷任主任": "/p/404-1120-53008.php?Lang=zh-tw",
    "系所介紹歷史沿革": "/p/404-1120-44081.php?Lang=zh-tw",
    "系所介紹現況報導": "/p/404-1120-44188.php?Lang=zh-tw",
    "系所介紹研究目標": "/p/404-1120-44080.php?Lang=zh-tw",
    "系所介紹願景與教學目標": "/p/404-1120-44190.php?Lang=zh-tw",
    "系所位置": "/p/404-1120-44191.php?Lang=zh-tw",
    "研究中心資管所": "/p/404-1120-44085.php?Lang=zh-tw",
    "研究中心醫資所": "/p/404-1120-44084.php?Lang=zh-tw",
    "學分學程": "/p/404-1120-25378.php?Lang=zh-tw",
    "SAS課程": "/p/404-1120-25541.php?Lang=zh-tw",
    "系友交流": "/p/412-1120-3558.php?Lang=zh-tw",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def crawl_page(name, path):
    url = BASE_URL + path
    print(f"爬取中：{name} ({url})")
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        
        # 自動偵測編碼
        import chardet
        detected = chardet.detect(res.content)
        encoding = detected.get("encoding", "utf-8")
        print(f"偵測到編碼：{encoding}")
        
        content = res.content.decode(encoding, errors="ignore")
        soup = BeautifulSoup(content, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [line for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        return clean_text

    except Exception as e:
        print(f"錯誤：{name} 爬取失敗 - {e}")
        return ""

def save_text(name, text):
    os.makedirs("docs", exist_ok=True)
    filepath = f"docs/{name}.txt"
    # 強制用 utf-8-sig 存檔（Windows 相容）
    with open(filepath, "w", encoding="utf-8-sig") as f:
        f.write(text)
    print(f"已儲存：{filepath}（{len(text)} 字）")

def download_and_read_pdf(url, name):
    try:
        print(f"下載PDF：{url}")
        res = requests.get(url, headers=HEADERS, timeout=30)
        if res.status_code == 200:
            pdf_path = f"docs/{name}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(res.content)
            
            # 用 pymupdf 讀取文字
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            if text.strip():
                save_text(name, text)
                print(f"PDF 解析成功：{name}")
            else:
                print(f"PDF 無法解析文字（可能是掃描版）：{name}")
    except Exception as e:
        print(f"PDF 下載失敗：{name} - {e}")

def crawl_pdfs_from_page(page_url):
    try:
        res = requests.get(page_url, headers=HEADERS, timeout=10)
        detected = chardet.detect(res.content)
        encoding = detected.get("encoding", "utf-8")
        content = res.content.decode(encoding, errors="ignore")
        soup = BeautifulSoup(content, "html.parser")
        
        # 找所有 PDF 連結
        pdf_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith(".pdf"):
                if href.startswith("http"):
                    full_url = href
                else:
                    full_url = BASE_URL + href
                name = a.get_text(strip=True) or href.split("/")[-1]
                name = re.sub(r'[\\/*?:"<>|]', '', name)[:50]
                pdf_links.append((name, full_url))
        
        return pdf_links
    except Exception as e:
        print(f"爬取PDF連結失敗：{e}")
        return []

def run_crawler():
    print("開始爬取系所網站...")
    for name, path in PAGES.items():
        text = crawl_page(name, path)
        if text:
            save_text(name, text)
        time.sleep(1)
    
    print("\n開始下載PDF文件...")
    pdf_pages = [
        BASE_URL + "/p/412-1120-3300.php?Lang=zh-tw",  # 學士班
        BASE_URL + "/p/412-1120-3301.php?Lang=zh-tw",  # 資管所
        BASE_URL + "/p/412-1120-3333.php?Lang=zh-tw",  # 表單學士
        BASE_URL + "/p/412-1120-3334.php?Lang=zh-tw",  # 表單資管所
    ]
    
    for page_url in pdf_pages:
        pdf_links = crawl_pdfs_from_page(page_url)
        for name, url in pdf_links:
            download_and_read_pdf(url, name)
            time.sleep(1)
    
    print("\n全部完成！所有文件已存入 docs/ 資料夾")

if __name__ == "__main__":
    run_crawler()