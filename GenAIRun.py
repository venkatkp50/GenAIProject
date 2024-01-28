import pandas as pd
from heapq import nlargest
import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from supportDef import highlightText
from sentTrans import bert_model
from datetime import date
from keyWord import getKeyWords
# from plots import getPlot
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
label_dict = {'General': 0, 'Bookmarks & History': 1, 'Untriaged': 2, 'Tabbed Browser': 3, 'Developer Tools': 4, 'Toolbars and Customization': 5, 'Preferences': 6, 'Location Bar': 7, 'Menus': 8, 'Theme': 9, 'Shell Integration': 10, 'File Handling': 11, 'Installer': 12, 'Extension Compatibility': 13, 'Session Restore': 14, 'Security': 15, 'Search': 16, 'Panorama': 17, 'Build Config': 18, 'Keyboard Navigation': 19, 'Private Browsing': 20, 'SocialAPI': 21, 'Disability Access': 22, 'Phishing Protection': 23, 'RSS Discovery and Preview': 24, 'PDF Viewer': 25, 'Migration': 26, 'Page Info Window': 27, 'Downloads Panel': 28, 'Help Documentation': 29, 'Web Apps': 30, 'Microsummaries': 31, 'Webapp Runtime': 32, 'SocialAPI: Providers': 33, 'Shumway': 34, 'WinQual Reports': 35}
st.image('images/Banner2.png')


# df = pd.read_csv('Mozzila_Bug.csv')
df1 = pd.read_csv('mozilla_firefox1.csv')
df21 = pd.read_csv('mozilla_firefox23.csv')
df22 = pd.read_csv('mozilla_firefox23.csv')
df23 = pd.read_csv('mozilla_firefox23.csv')
df = pd.concat([df21,df1,df22,df23])
st.subheader('Dataset Sample')
st.dataframe(df[['Title','Description','Component']].head(5))
st.divider()

with st.sidebar:
    st.divider()
    st.subheader('Mozzilla Firefox Bug Dataset')
    st.divider()
    st.sidebar.dataframe(df['Component'].value_counts())


data_text = st.text_input('Enter Search String', '')
# data_text = data_text.replace('Example :','')
data = pd.Series( data_text)

count_vect = pickle.load(open("vectorizer.pickle", 'rb'))
tfidf_transformer = pickle.load(open("tfidfvectorizer.pickle", 'rb'))
dbfile = open('Pickle_LR_Model.pkl', 'rb')
# bert_model = pickle.load(open("bert_model.pkl", 'rb'))

def getLabelVal(text):
  return label_dict[text]
def getLabelName(kval):
  return list(label_dict.keys())[list(label_dict.values()).index(kval)]
def getTicketNo(id):
  return df.iloc[id]['Issue_id']
  

if st.button('Predict Business Service Group'):
    if not len(data_text) == 0:
        X_new_counts = count_vect.transform(data)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        model_lr = pickle.load(dbfile)
        predicted_pickle = model_lr.predict(X_new_tfidf)
        pred_int_label = getLabelName(predicted_pickle[0])
    
    st.divider()
    st.subheader('Predicted Business Group')
    st.success(pred_int_label)
    
    max_date = date(2014, 1, 31)
    st.dataframe(df.head(4))
    df['Component_new'] = df['Component'].replace(['Developer Tools', 'Developer Tools: Console','Developer Tools: Inspector','Developer Tools: Graphic Commandline and Toolbar','Developer Tools: Scratchpad', 'Developer Tools: Netmonitor','Developer Tools: Source Editor', 'Developer Tools: Debugger','Developer Tools: Style Editor', 'Developer Tools: Framework','Developer Tools: 3D View', 'Developer Tools: Object Inspector','Developer Tools: Responsive Mode', 'Developer Tools: Profiler','Developer Tools: App Manager','Developer Tools: WebGL Shader Editor', 'Developer Tools: Memory'], 'Developer Tools')
    df['int_label'] = df['Component_new'].apply(getLabelVal)
    df = df.dropna(subset=['Description'])
    st.dataframe(df.head(4))
    # df = df[((df['int_label']==predicted_pickle[0]) & (df['Status']=='RESOLVED') & (df['Resolution']!='INCOMPLETE') & (df['Resolution'] !='DUPLICATE') & ((pd.to_datetime(max_date,utc=True) - pd.to_datetime(df['Resolved_time'],utc=True,infer_datetime_format=True)).dt.days < 600) )]
    df = df[((df['int_label']==predicted_pickle[0]) & (df['Status']=='RESOLVED') & (df['Resolution']!='INCOMPLETE') & (df['Resolution'] !='DUPLICATE'))]
    
    st.divider()

    Sentences = df['Description'].tolist()
    sentence_vecs = bert_model.encode(Sentences)
    data_vecs = bert_model.encode(data)
    similiarity_array = cosine_similarity(data_vecs, sentence_vecs[:])
    similiarity_list  = similiarity_array.tolist()
    similiarity_dict = {}
    
    cnt = 0
    for item in similiarity_list[0]:
        similiarity_dict[cnt] = item
        cnt +=1
    res = nlargest(5, similiarity_dict, key=similiarity_dict.get)
    sim_df = []
    body_df = []
    resolution_df = []
    keyword_df = []
    for x in res:
       sim_df.append(str(round(similiarity_dict[x]*100,1))+'%')
       body_df.append(df.iloc[x]['Description'])
       resolution_df.append(df.iloc[x]['Resolution'])
       keyword_df.append(getKeyWords(df.iloc[x]['Description']))


    output = pd.DataFrame({'Issue ID':res,'Percent Match':sim_df,'Resolution':resolution_df,'Key Words':keyword_df})
    st.subheader('Matching Cases')
    st.dataframe(output.set_index(output.columns[0]))
st.divider()
