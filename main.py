# This is a sample Python script.
import os

import openai

from fine_tuning import tuning
from web_crawler import crawl, answer_question, print_one


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

openai.api_key = "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"
# openai.Model.list()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("=================【start】=================")
    # crawl("https://openai.com/")
    print_one("天津天气怎么样")
    # crawl("https://openai.com/")
    # tuning()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
