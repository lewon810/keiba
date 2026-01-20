import requests

def fetch_and_save(url, filename="debug.html"):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Encoding: {response.encoding}, Apparent: {response.apparent_encoding}")
        
        response.encoding = response.apparent_encoding if response.apparent_encoding else response.encoding
        
        content = response.text
        print(f"Content Length: {len(content)}")
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved {url} to {filename}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_and_save("https://race.netkeiba.com/race/shutuba.html?race_id=202606010809&rf=race_list")
