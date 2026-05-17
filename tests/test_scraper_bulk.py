import pytest
from bs4 import BeautifulSoup
from train.scraper_bulk import scrape_race_data

def test_scrape_race_data_column_parsing(monkeypatch):
    # Mock fetch_html to return an old (21 columns) HTML table
    old_html = """
    <html>
      <dl class="racedata">
        <dd><p><span>芝右1600m / 天候 : 晴 / 芝 : 良 / 発走 : 09:50</span></p></dd>
        <p class="smalltxt">2016年1月5日 1回中山1日目</p>
      </dl>
      <table class="race_table_01">
        <tr><th>header</th></tr>
        <tr>
          <td>1</td><td>1</td><td>1</td><td><a href="/horse/123/">Horse1</a></td><td>牡3</td><td>55</td>
          <td><a href="/jockey/123/">Jockey1</a></td><td>1:35.0</td><td>-</td><td>-</td><td>-</td>
          <td>34.5</td><td>1.5</td><td>1</td><td>480(+2)</td><td>-</td><td>-</td><td>-</td><td><a href="/trainer/123/">Trainer1</a></td><td>-</td><td>-</td>
        </tr>
      </table>
    </html>
    """
    
    # Mock fetch_html to return a new (25 columns) HTML table
    new_html = """
    <html>
      <dl class="racedata">
        <dd><p><span>ダ左1400m / 天候 : 晴 / ダート : 良 / 発走 : 10:05</span></p></dd>
        <p class="smalltxt">2024年1月27日 1回東京1日目</p>
      </dl>
      <table class="race_table_01">
        <tr><th>header</th></tr>
        <tr>
          <td>1</td><td>4</td><td>7</td><td><a href="/horse/456/">Horse2</a></td><td>牡3</td><td>57</td>
          <td><a href="/jockey/456/">Jockey2</a></td><td>1:26.9</td><td>-</td>
          <td>**</td><td>**</td><td>**</td><td>**</td><td>**</td>
          <td>4-4</td><td>37.3</td><td>1.4</td><td>1</td><td>480(+2)</td>
          <td>**</td><td>**</td><td>**</td><td><a href="/trainer/456/">Trainer2</a></td><td>-</td><td>-</td>
        </tr>
      </table>
    </html>
    """
    
    def mock_fetch(url):
        if "2016" in url:
            return old_html
        return new_html
        
    monkeypatch.setattr("train.scraper_bulk.fetch_html", mock_fetch)
    
    # Test Old format
    res_old = scrape_race_data("201601010101")
    assert res_old is not None
    assert len(res_old) == 1
    assert res_old[0]["horse_weight"] == "480"
    assert res_old[0]["weight_diff"] == "+2"
    assert res_old[0]["odds"] == "1.5"
    assert res_old[0]["popularity"] == "1"
    assert res_old[0]["trainer_id"] == "123"
    
    # Test New format
    res_new = scrape_race_data("202401010101")
    assert res_new is not None
    assert len(res_new) == 1
    assert res_new[0]["horse_weight"] == "480"
    assert res_new[0]["weight_diff"] == "+2"
    assert res_new[0]["odds"] == "1.4"
    assert res_new[0]["popularity"] == "1"
    assert res_new[0]["trainer_id"] == "456"
