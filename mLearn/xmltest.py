#!/usr/bin/env python
#coding:utf-8
"""
__title__ = ""
__author__ = "adw"
__mtime__ = "2016/7/17"
__purpose__ = 
"""
import re

from bs4 import BeautifulSoup, element

# with open(r"F:\17MedPro\workspace\medstdy\docs\template_with_data_n.xml", "rb") as f:
#     cont = f.readlines()

# new = [x.strip("\n") for x in cont]
#
# with open(r"F:\17MedPro\workspace\medstdy\docs\template_with_data_n.xml", "w") as f:
#     f.writelines(new)

soup = BeautifulSoup(open(r"F:\17MedPro\workspace\medstdy\docs\template_with_data.xml"), "xml")
# print soup.find_all("user_id")
tbl_list = soup.find_all(re.compile(r"tbl_clinical_course"))
for tbl in tbl_list:
    print tbl.name
    fields = tbl.children
    doc = {}
    for field in fields:
        print type(field)
        if isinstance(field, element.Tag):
            items = field.contents
            if len(items) > 1:
                item_list = []
                for item in items:
                    if not isinstance(item, element.Tag):
                        continue
                    item_doc = {}
                    for it in item:
                        if not isinstance(it, element.Tag):
                            continue
                        item_doc[it.name.encode("utf-8")] = it.text.encode("utf-8").strip()
                    item_list.append(item_doc)
                doc[field.name.encode("utf-8")] = item_list

                print "ok"
            print field.name
            print field.text

pass

# soup = BeautifulSoup(open("NCT00005144.xml"), "xml")
# list = soup.item
# for x in list:
#     print x.name
#     # print x
# print list.contents
pass