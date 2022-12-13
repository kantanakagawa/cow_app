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

class Sucreiping:
    
    def get_year(self):
        l = []
        next_link = driver.find_elements_by_class_name("schedule-archive__item")
        for i in next_link:
            atag = i.find_element_by_tag_name("a")
            l.append(atag.get_attribute("href"))
        return l
    def cow_scraping(self, url):
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
        return url_list[3::4]
    def cow_scraping_2020(self, url):
        driver.get(url)
        cow_list = []
        for elem_span in driver.find_elements_by_xpath('//a/span'):
            sleep(1)
            elem_a = elem_span.find_element_by_xpath('..')
            if elem_span.text == "和牛子牛黒毛和種" :
                cow_list.append(elem_a.get_attribute('href'))
        cow_list = cow_list[4:]
        url_list = []
        for cow_link in cow_list:
            sleep(1)
            driver.get(cow_link)
            element_class = driver.find_elements_by_class_name("single-schedule__data")
            for i in element_class:
                atag = i.find_element_by_tag_name("a")
                url_list.append(atag.get_attribute("href"))
        return url_list[3::4]
    def extract(self, filepath):
        _df=pd.read_excel(filepath)
        columns=_df.iloc[1,[2,3,4,5,6,7,8,9]]
        df=_df.iloc[2:len(_df),[2,3,4,5,6,7,8,9]]
        df.columns=columns
        dating = _df.columns[1]
        d = dating.find("\u3000")
        dd = _df.columns[1][:d]
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
    def get_data(self):
        l = self.get_year()
        l_0 = l[0]
        l_1 = l[1]
        l_2 = l[2]
        xlsx_list_2022 = self.cow_scraping(l_0)
        xlsx_list_2021 = self.cow_scraping(l_1)
        xlsx_list_2020 = self.cow_scraping_2020(l_2)
        all_xlsx_list = xlsx_list_2022+xlsx_list_2021+xlsx_list_2020
        df=pd.DataFrame()
        for filepath in all_xlsx_list:
            _df=self.extract(filepath)
            df=pd.concat([df,_df])
        df.dropna()
        df=df.reset_index(drop=True)
        d = len(df["年月日"])
        for i in range(d):
            df["年月日"][i] = self.japanese_calendar_converter(df["年月日"][i])
        df["年月日"] = pd.to_datetime(df["年月日"])
        column = df.columns[5:8]
        for col in column:
            df[col] = df[col].astype('int')
        driver.close()
        return df