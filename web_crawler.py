import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from ast import literal_eval

openai.api_key = "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"
openai.Model.list()

HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

# Define root domain to crawl
domain = "openai.com"
full_url = "https://openai.com/"

# 创建一个类来解析HTML并获取超链接
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # 创建一个列表来存储超链接
        self.hyperlinks = []

    # 重写HTMLParser的handle_starttag方法来获取超链接
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # 具有href属性，则将href属性添加到超链接列表中
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])


### step 2

# 从url获取超链接
def get_hyperlinks(url):
    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []

            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

### step 3
# 函数从同一域中的URL获取超链接
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif (
                link.startswith("#")
                or link.startswith("mailto:")
                or link.startswith("tel:")
            ):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))

### step 4
#
def crawl(url):
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the text files
    if not os.path.exists("text/"):
        os.mkdir("text/")

    if not os.path.exists("text/" + local_domain + "/"):
        os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
        os.mkdir("processed")

    # While the queue is not empty, continue crawling
    while queue:

        # Get the next URL from the queue
        url = queue.pop()
        print("url: " + url)  # for debugging and to see the progress

        # Save text from the url to a <url>.txt file
        with open('text/' + local_domain + '/' + url[8:].replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:

            # Get the text from the URL using BeautifulSoup
            soup = BeautifulSoup(requests.get(url).text, "html.parser")

            # Get the text but remove the tags
            text = soup.get_text()

            # If the crawler gets to a page that requires JavaScript, it will stop the crawl
            if ("You need to enable JavaScript to run this app." in text):
                print("Unable to parse page " + url + " due to JavaScript being required")

            # Otherwise, write the text to the file in the text directory
            f.write(text)

        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)


# crawl(full_url)

# ### step 5
def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie
#
# ### Step 6
# # Create a list to store the text files
texts=[]
#
# # 获取文本目录中的所有文本文件
for file in os.listdir("text/" + domain + "/"):

    # Open the file and read the text
    with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
        text = f.read()
        # print("text: ", text)

        # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
        texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))
#
# # Create a dataframe from the list of texts
df = pd.DataFrame(texts, columns = ['fname', 'text'])
#
# 将文本列设置为去掉换行符的原始文本
df['text'] = df.fname + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')
df.head()
#
# ### step 7
# 加载用于ada-002模型的cl100k_base标记器
tokenizer = tiktoken.get_encoding("cl100k_base")
#
df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']
#
# # 对文本进行标记，并将标记的数量保存到一个新列中对文本进行标记，并将标记的数量保存到一个新列中
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
#
# 使用直方图可视化每行令牌数量的分布
df.n_tokens.hist()
max_tokens = 500
# ### step 8
# 最新的嵌入模型可以处理最多 8191 个输入标记的输入，因此大多数行不需要任何分块，
# 但对于每个刮取的子页面来说可能并非如此，因此下一个代码块会将较长的行拆分为较小的块
def split_into_many(text, max_tokens=max_tokens):
    # 把文章分成句子
    sentences = text.split('. ')

    # 获取每个句子的标记数获取每个句子的标记数
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # 循环遍历元组中连接在一起的句子和标记循环遍历元组中连接在一起的句子和标记
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks
#
shortened = []
#
# # Loop through the dataframe
for row in df.iterrows():

    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])

    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append(row[1]['text'])
#
# ### step 9
df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
# print("size: ", df.values.size)
df.n_tokens.hist()
#
#
# ### step 10
# # Note that you may run into rate limit issues depending on how many files you try to embed
# # Please check out our rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits
# 速率限制 fillna 替换nan值
df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
df.to_csv('processed/embeddings.csv')
df.head()
#
# ### step 11
df=pd.read_csv('processed/embeddings.csv', index_col=0)
# print("size: ", df['embeddings'].size)
# print("df: ", df['embeddings'].dropna())
df['embeddings'] = df['embeddings'].fillna({"k": "v"}).apply(literal_eval).apply(np.array).dropna()

df.head()
# ### step 12
# 将问题转换为具有简单函数的嵌入。这很重要，因为嵌入搜索使用余弦距离比较数字向量（这是原始文本的转换）。
# 这些向量可能是相关的，如果它们的余弦距离接近，则可能是问题的答案
def create_context(
        question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # 获取问题嵌入
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # 得到嵌入的距离
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # 按距离排序并将文本添加到上下文中，直到上下文太长按距离排序并将文本添加到上下文中，直到上下文太长
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # 将文本的长度添加到当前长度
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)
#
#
def answer_question(
        df,
        model="text-davinci-003",
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=150,
        stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    # if debug:
    #     print("Context:\n" + context)
    #     print("\n\n")

    try:
        # 使用问题和上下文创建一个完成题使用问题和上下文创建一个context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""
#
#
def print_one(str):
    print("zzh openAI kspay回答：", answer_question(df, question=str, debug=False))

if __name__ == '__main__':
    print_one("今天多少度？")
#
# print(answer_question(df, question="What is our newest embeddings model?"))