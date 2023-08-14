%%writefile hello/hello.py

import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models import FastText
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity
import requests
import gensim
from flask import Flask, request, jsonify
import urllib.request
import pymysql
import csv
from flasgger import Swagger

app = Flask(__name__)



##1. 대외활동
conn = pymysql.connect(host='hisujung-db.cuidqegkqifp.ap-northeast-2.rds.amazonaws.com', port=3306, user='admin', passwd='hisujung', db='hisujungDB', charset='utf8')

try:
    cursor = conn.cursor()
    sql = "SELECT DISTINCT title FROM ExternalAct LIMIT 100;"
    cursor.execute(sql)
    result = cursor.fetchall()

    # Writing the data to a CSV file locally on the client-side
    with open('contest.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([i[0] for i in cursor.description])  # Write column headers
        csv_writer.writerows(result)  # Write data rows

    print("Data exported to contest.csv successfully.")
except Exception as e:
    print("Error:", e)
finally:
    conn.close()
    
## 2. 교내활동
conn2 = pymysql.connect(host='hisujung-db.cuidqegkqifp.ap-northeast-2.rds.amazonaws.com', port=3306, user='admin', passwd='hisujung', db='hisujungDB', charset='utf8')

try:
    cursor2 = conn2.cursor()
    sql2 = "SELECT DISTINCT title FROM UnivActivity LIMIT 100;"
    cursor2.execute(sql2)
    result2 = cursor2.fetchall()

    # Writing the data to a CSV file locally on the client-side
    with open('univ.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([i[0] for i in cursor2.description])  # Write column headers
        csv_writer.writerows(result2)  # Write data rows

    print("Data exported to univ.csv successfully.")
except Exception as e:
    print("Error:", e)
finally:
    conn2.close()
    

##1 교내!
# 데이터를 데이터프레임으로 로드
df = pd.read_csv("univ.csv")
okt = Okt()
tokenized_data = []
for sentence in df['title']:
    tokenized_sentence = okt.morphs(sentence, stem=True)
    tokenized_data.append(tokenized_sentence)

# FastText 모델 학습
model = FastText(vector_size=200, window=5, min_count=2, workers=-1)
model.build_vocab(corpus_iterable=tokenized_data)
model.train(corpus_iterable=tokenized_data, total_examples=len(tokenized_data), epochs=15)
word_vectors = model.wv
activities = df[['title']]

# 문장 벡터 계산
sentence_vectors = [word_vectors[tokenized_sentence].mean(axis=0) for tokenized_sentence in tokenized_data]
cosine_similarities = cosine_similarity(sentence_vectors, sentence_vectors)
    
    
##2 대외!
# 데이터를 데이터프레임으로 로드
dfE = pd.read_csv("contest.csv")
oktE = Okt()
tokenized_dataE = []
for sentence in dfE['title']:
    tokenized_sentenceE = oktE.morphs(sentence, stem=True)
    tokenized_dataE.append(tokenized_sentenceE)

# FastText 모델 학습
modelE = FastText(vector_size=200, window=5, min_count=2, workers=-1)
modelE.build_vocab(corpus_iterable=tokenized_dataE)
modelE.train(corpus_iterable=tokenized_dataE, total_examples=len(tokenized_dataE), epochs=15)
word_vectorsE = modelE.wv
activitiesE = dfE[['title']]

# 문장 벡터 계산
sentence_vectorsE = [word_vectorsE[tokenized_sentenceE].mean(axis=0) for tokenized_sentenceE in tokenized_dataE]
cosine_similaritiesE = cosine_similarity(sentence_vectorsE, sentence_vectorsE)

@app.route("/recommend/univ", methods=['GET'])
def univAct():
    univ_activity_id = request.args.get('activity_id', type=int, default='')
        # 입력한 ID에 해당하는 제목 찾기
    title =df['title'][univ_activity_id]

    # 유사한 활동 5개 선정
    indices= pd.Series(df.index, index=df['title']).drop_duplicates(keep='first')
    if title in indices:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_similarities[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        activity_indices = [i[0] for i in sim_scores]
        recommend_df = activities.iloc[activity_indices].reset_index(drop=True)
    # Assuming 'recommend' is the result of the recommendations

    # Extract the titles from the recommendation dataframe
    recommend_titles = recommend_df['title'].tolist()

    # Create placeholders for the SQL query
    placeholders = ', '.join(['%s'] * len(recommend_titles))

    # SQL query to fetch information based on titles
    sql = f"SELECT univ_activity_id, title, link FROM UnivActivity WHERE title IN ({placeholders});"

    conn = pymysql.connect(host='hisujung-db.cuidqegkqifp.ap-northeast-2.rds.amazonaws.com', port=3306, user='admin', passwd='hisujung', db='hisujungDB', charset='utf8')
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, recommend_titles)
            result = cursor.fetchall()
    finally:
        conn.close()
        # Create a dataframe from the SQL query result
    result_df = pd.DataFrame(result, columns=['univ_activity_id', 'title', 'link'])
    return jsonify({'reccomend': result_df.to_dict(orient='records')})
    
@app.route("/recommend/external", methods=['GET'])
def externalAct():
    external_activity_id = request.args.get('activity_id', type=int, default='')
        # 입력한 ID에 해당하는 제목 찾기
    title =dfE['title'][external_activity_id]

    # 유사한 활동 5개 선정
    indices= pd.Series(df.index, index=dfE['title']).drop_duplicates(keep='first')
    if title in indices:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_similaritiesE[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        activity_indices = [i[0] for i in sim_scores]
        recommend_df = activitiesE.iloc[activity_indices].reset_index(drop=True)

    # Extract the titles from the recommendation dataframe
    recommend_titles = recommend_df['title'].tolist()

    # Create placeholders for the SQL query
    placeholders = ', '.join(['%s'] * len(recommend_titles))

    # SQL query to fetch information based on titles
    sql = f"SELECT external_act_id, title, link FROM ExternalAct WHERE title IN ({placeholders});"

    conn = pymysql.connect(host='hisujung-db.cuidqegkqifp.ap-northeast-2.rds.amazonaws.com', port=3306, user='admin', passwd='hisujung', db='hisujungDB', charset='utf8')
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, recommend_titles)
            result = cursor.fetchall()
    finally:
        conn.close()
        # Create a dataframe from the SQL query result
    result_df = pd.DataFrame(result, columns=['external_act_id', 'title', 'link'])

    return jsonify({'reccomend': result_df.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)
    
