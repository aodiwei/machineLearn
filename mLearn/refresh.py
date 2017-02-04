import requests
from multiprocessing import dummy
# import mechanize
import cookielib
import time



def vist_room(num):
    """

    :return:
    """
    respones = requests.get("http://www.yy.com/223987")

    print(num)
    print(respones)


def mock_brower(num):
    print(num)
    # Browser
    br = mechanize.Browser()
    # Cookie Jar
    # cj = cookielib.LWPCookieJar()
    # br.set_cookiejar(cj)
    # Browser options
    br.set_handle_equiv(True)
    br.set_handle_gzip(True)
    br.set_handle_redirect(True)
    br.set_handle_referer(True)
    br.set_handle_robots(False)
    # Follows refresh 0 but not hangs on refresh > 0
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)
    # Want debugging messages?
    # br.set_debug_http(True)
    # br.set_debug_redirects(True)
    # br.set_debug_responses(True)
    # User-Agent (this is cheating, ok?)
    br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
    r = br.open('https://www.douyu.com/1127295')
    r.read()
    print r


import unittest
from selenium import webdriver
from bs4 import BeautifulSoup
import os
class seleniumTest(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.PhantomJS(executable_path="/Users/David/PycharmProjects/phantomjs/bin/phantomjs")

    def testEle(self):
        driver = self.driver
        driver.get('https://www.douyu.com/1183774')
        # print driver.page_source



    def tearDown(self):
        print 'down'

def selenium(num):
    unittest.main()
    unittest.main()
    unittest.main()
    unittest.main()
    unittest.main()
    unittest.main()
    time.sleep(1)
    print num


def phant(num):
    # driver = webdriver.PhantomJS(executable_path="/Users/David/PycharmProjects/phantomjs/bin/phantomjs")
    # driver = webdriver.Chrome(executable_path="/Users/David/PycharmProjects/sometest/chromedriver")
    # page = driver.get('http://blog.csdn.net/yinwenjie/article/details/53407288d')
    for i in range(0, 1):
        # driver = webdriver.PhantomJS(executable_path="/Users/David/PycharmProjects/phantomjs/bin/phantomjs")

        # driver.get('http://blog.csdn.net/yinwenjie/article/details/53407288')
        # option = webdriver.ChromeOptions()
        # option.add_argument('--user-data-dir=C:\Users\Administrator\AppData\Local\Google\Chrome\User Data')
        # driver = webdriver.Chrome(executable_path="/Users/David/PycharmProjects/sometest/chromedriver", chrome_options=option)
        # driver = webdriver.Chrome(executable_path=r"C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe")
        # driver.get('http://www.yy.com/59940336/59940336')

        chromedriver = "D:\DownLoad\chromedriver_win32\chromedriver.exe"
        os.environ["webdriver.chrome.driver"] = chromedriver

        driver = webdriver.Chrome(chromedriver)
        driver.get('http://www.yy.com/59940336/59940336')
        time.sleep(45)
        print driver.session_id
        driver.stop_client()
        # driver.refresh()
        # driver.close()


        # driver.get('https://www.douyu.com/1183774')
        # driver.get('https://www.douyu.com/1183774')
        # driver.get('https://www.douyu.com/1183774')
        # driver.get('https://www.douyu.com/1183774')
        # driver.get('https://www.douyu.com/1183774')
        print i

if __name__ == '__main__':
    # pool = dummy.Pool(10)
    # pool.map(vist_room, range(0, 100))
    for num in range(0, 100):
        vist_room(num)
        time.sleep(1)
    # phant(1)http://www.yy.com/59940336/5994033
    # map(phant, range(0, 1))