import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data(filename):
    #数据评论集合 {书名：评论1+评论2+。。。}
    book_comments={}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        next(reader)  # 跳过表头
        for row in reader:
            book = row['book']
            comment = row['body']
            conment_words = jieba.lcut(comment)
            if book=='':continue #跳过空书名
            #评论集合收集
            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(conment_words)
    return book_comments

if __name__ == '__main__':
    #加载停用词表
    stopwords = [line.strip() for line in open('/Users/circleLee/Develop/workspace_py/nlp/week05/stopwords.txt', 'r', encoding='utf-8')]
    #加载数据
    book_comments = load_data('/Users/circleLee/Develop/workspace_py/nlp/week05/douban_comments_fixed.txt')
    print(len(book_comments))

    #提取书名和评论文本
    book_names=[]
    book_comms=[]
    for book,comments in book_comments.items():
        book_names.append(book)
        book_comms.append(comments)
    #构建TF-IDF特征矩阵
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix=vectorizer.fit_transform([''.join(comms) for comms in book_comms])

    #计算图书间的余弦相似度
    similarity = cosine_similarity(tfidf_matrix)
    #print(similarity)

    #推荐图书
    book_list=list(book_comments.keys())
    print(book_list)
    book_name=input("请输入图书名称：")
    #获取图书索引
    book_idx=book_names.index(book_name)
    
    #根据索引获取与该图书最相似的图书
    recommend_book_index=np.argsort(-similarity[book_idx])[1:11] #前10本
    #输出推荐的图书
    for idx in recommend_book_index:
        print(f"《{book_names[idx]}》\t 相似度：{similarity[book_idx][idx]:.4f}")
    print()