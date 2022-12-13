import chromedriver_binary
from selenium import webdriver
from time import sleep
import openpyxl
import pandas as pd
import unicodedata
import re
import datetime

driver = webdriver.Chrome()
driver.get('https://www.kenchikukyo.jp/schedule/')

class Sucraping_detail:

    def get_years(self):
        l = []
        next_link = driver.find_elements_by_class_name("schedule-archive__item")
        for i in next_link:
            atag = i.find_element_by_tag_name("a")
            l.append(atag.get_attribute("href"))
        return l

    def listing_scraping(self,url):
        driver.get(url)
        cow_list = []
        for elem_span in driver.find_elements_by_xpath("//a/span"):
            sleep(1)
            elem_a = elem_span.find_element_by_xpath("..")
            if elem_span.text == "和牛子牛黒毛和種":
                cow_list.append(elem_a.get_attribute("href"))
        url_list = []
        for cow_link in cow_list:
            sleep(1)
            driver.get(cow_link)
            element_class = driver.find_elements_by_class_name("single-schedule__data")
            for i in element_class:
                atag = i.find_element_by_tag_name("a")
                url_list.append(atag.get_attribute("href"))
        return url_list

    def make_nkb_list(self,year_urls):
        url_list = []
        for i in year_urls:
            url = self.listing_scraping(i)
            url_list.append(url)

        nkb_url = []

        for url_lists in url_list:
            for i in url_lists:
                if 0 < i.find("NKB.xlsx"):
                    if 0>i.find("KDNKB.xlsx"):
                        nkb_url.append(i)

        bull_list = ['https://www.kenchikukyo.jp/wp-content/uploads/2020/09/R020910NKB.xlsx','https://www.kenchikukyo.jp/wp-content/uploads/2020/09/R020911NKB.xlsx','https://www.kenchikukyo.jp/wp-content/uploads/2020/08/R020807NKB.xlsx','https://www.kenchikukyo.jp/wp-content/uploads/2020/08/R020806NKB.xlsx','https://www.kenchikukyo.jp/wp-content/uploads/2020/07/R020710NKB.xlsx','https://www.kenchikukyo.jp/wp-content/uploads/2020/07/R020709NKB.xlsx','https://www.kenchikukyo.jp/wp-content/uploads/2020/06/R020612NKB.xlsx','https://www.kenchikukyo.jp/wp-content/uploads/2020/06/R020611NKB.xlsx','https://www.kenchikukyo.jp/wp-content/uploads/2020/05/R020515NKB.xlsx','https://www.kenchikukyo.jp/wp-content/uploads/2020/05/R020514NKB.xlsx',]

        for i in bull_list:
            index = nkb_url.index(i)
            nkb_url.pop(index)
        return nkb_url

    def extrac(self,file_path):
        _df=pd.read_excel(file_path)
        df = _df.iloc[3:,1:27]
        df.columns = _df.iloc[2,1:27]
        d = _df.iloc[0][1]
        aa = d.find("月")
        bb = d.find("月",aa+1)
        dd = d[:aa] + d[bb:-5]
        df = df.assign(年月日=dd)
        return df
    
# 各年号の元年を定義
    def japanese_calendar_converter(self,text):
        eraDict = {
            "明治": 1868,
            "大正": 1912,
            "昭和": 1926,
            "平成": 1989,
            "令和": 2019,
        }


        # 正規化
        normalized_text = unicodedata.normalize("NFKC", text)

        # 年月日を抽出
        pattern = r"(?P<era>{eraList})(?P<year>[0-9]{{1,2}}|元)年(?P<month>[0-9]{{1,2}})月(?P<day>[0-9]{{1,2}})日".format(eraList="|".join(eraDict.keys()))
        date = re.search(pattern, normalized_text)

        # 抽出できなかったら終わり
        if date is None:
            print("Cannot convert to western year")

        # 年を変換
        for era, startYear in eraDict.items():
            if date.group("era") == era:
                if date.group("year") == "元":
                    year = eraDict[era]
                else:
                    year = eraDict[era] + int(date.group("year")) - 1

        # date型に変換して返す
        return datetime.date(year, int(date.group("month")), int(date.group("day")))

    def get_detail_data(self):
        year_urls = self.get_years()
        nkb_list = self.make_nkb_list(year_urls)
        df=pd.DataFrame()
        for file_path in nkb_list:
            _df=self.extrac(file_path)
            df=pd.concat([df,_df])
        df=df.reset_index(drop=True)
        d = len(df["年月日"])
        for i in range(d):
            df["年月日"][i] = self.japanese_calendar_converter(df["年月日"][i])
        df["年月日"] = pd.to_datetime(df["年月日"])
        df = df.replace("2021-05-13","2021-05-27")
        df = df.replace("2021-05-14","2021-05-28")
        df["日令"] = df["日令"].astype('int')
        driver.close()
        return df