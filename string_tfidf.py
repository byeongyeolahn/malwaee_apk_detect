from androguard.core.bytecodes.apk import APK
from androguard.core.bytecodes.dvm import DalvikVMFormat
from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd
import numpy as np

# androguard 를 사용하여 문자열 추출
def androguard_string(filename):
    apkf = APK(filename)
    classes = apkf.get_dex()
    a = DalvikVMFormat(classes)
    return a.get_strings()

#디렉토리 내 존재하는 APK 파일 리스트화
def file_list_fun(file_path):
    apk_file = os.listdir(file_path)
    return apk_file

# tf-idf 계산
def tf_idf(word_list):
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(word_list)
    
    # 단어 출현 빈도 체크
    tf = pd.DataFrame(dtm.toarray(), columns = vectorizer.get_feature_names())
    df = tf.astype(bool).sum(axis=0)
    
    D = len(tf) # 단어의 개수
    idf = np.log((D+1) / (df+1)) + 1
    
    tfidf = tf * idf
    tfidf = tfidf / np.linalg.norm(tfidf, axis = 1, keepdims = True)
    return tfidf
    
if __name__ == "__main__":
    apk_dir_path = 'APK 파일 존재하는 디렉토리 경로'
    file_list = file_list_fun(apk_dir_path)
    print(tf_idf(androguard_string(apk_dir_path + '\\' + str(file_list[0]))))
