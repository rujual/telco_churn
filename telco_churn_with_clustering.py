import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# backend and sidebar-frontend code'

st.header('Customer Churn Telco')

#read data
@st.cache
def read_data(data_path):
    df1 = pd.read_csv(data_path)
    df1.replace(' ', '0', inplace=True)
    return df1


dfst = read_data('Data.csv')

df2 = dfst
df8 = df2

#read Clustering Centroids data
#Clustering already done in K-Means-Clustering.ipynb jupyter notebook in this same directory
@st.cache
def read_cent(data_path):
    dfc = pd.read_csv(data_path)
    return dfc

df_cent = read_cent("Data_centroids.csv")

# now we have trained random forest model- rfc_best as a imported pickle file
rfc_best = pickle.load(open('finalized_model.sav', 'rb'))


# Filtering Data
st.sidebar.header("Filter:")
df_cat = [x for x in dfst.columns if len(dfst[x].unique()) < 6]
df_cat.append('cluster_number')
df_num = list(set(dfst.columns) - set(df_cat))
grp = st.sidebar.multiselect("Select parameter(s) to Filter by: ", [x for x in df_cat])
df_temp = df8.copy(deep=True)

try:
    selected_params = {}
    l_df = []

    for x in range(len(grp)):
        d = {'key1': str(grp[x])}
        s = "{key1}:"
        s = s.format(**d)
        attri = st.sidebar.multiselect(label=s, options=dfst[(grp[x])].unique(), key=str(x))
        # st.write('selected',attri,type(attri))

        for y in range(0, len(attri)):
            if (df8.equals(df_temp)):
                df8 = df8[df8[(grp[x])] == attri[y]]
            else:
                df8 = pd.concat([df8, df_temp[df_temp[(grp[x])] == attri[y]]])
                df8.sort_index(inplace=True)
        df_temp = df8.copy(deep=True)
    sdfst = df8

except(IndexError):
    st.error("Select parameters Please!")

# Customer Search

st.sidebar.header('Customer Search:')
sbar = st.sidebar.text_input("Enter Customer ID:", key='sb')

# Uncomment these lines when running the code for the first time
# df_sear = dfst[dfst['customerID']=='7590-VHVEG']
# df_sear.to_csv("Searched_Records.csv", mode='w+')

df_sear = (pd.read_csv("Searched_Records.csv")).drop('Unnamed: 0', axis=1)
df_sear.to_csv("Searched_Records.csv", mode='w+')

if st.sidebar.button('Search'):
    st.subheader("Searched Record~")
    df_sear = dfst[dfst['customerID'] == sbar]
    df_sear.to_csv("Searched_Records.csv", mode='w+')
    st.dataframe(df_sear)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    my_dict = {'customerID': df_sear['customerID'].iloc[0], 'gender': df_sear['gender'].iloc[0],
               'SeniorCitizen': df_sear['SeniorCitizen'].iloc[0],
               'Partner': df_sear['Partner'].iloc[0], 'Dependents': df_sear['Dependents'].iloc[0]}
# Customized Use Case

st.sidebar.header('Customized Use-case:')
select_param = {}
count = 0
df_cat.remove('Churn')
df_cat1 = df_cat
l12 = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

for x in l12:
    df_cat1.remove(x)

# temporary changes for presentation
df_cat1 = ['InternetService', 'OnlineSecurity']

for x in df_cat1:
    d = {'key1': str(x)}
    s = "Select {key1}:"
    s = s.format(**d)
    l_unique = list(dfst[x].unique())
    l_ind = l_unique.index(str(df_sear[x].iloc[0]))
    temp = st.sidebar.selectbox(label=s, index=l_ind, options=l_unique,
                                key=str(x))  # index=(l_unique.index(df_sear[x])),
    select_param[x] = temp

df_num1 = df_num
df_num1.remove('customerID')

# temporary changes for presentation

df_num1 = ['tenure', 'TotalCharges']

for y in df_num1:
    d = {'key1': str(y)}
    s = "Set {key1}:"
    s = s.format(**d)
    temp1 = st.sidebar.slider(label=s, min_value=(dfst[y].unique()).astype(float).min(),
                              max_value=(dfst[y].unique()).astype(float).max(),
                              value=float(df_sear[y]),
                              step=(((dfst[y].unique()).astype(float).max()) - int(
                                  (dfst[y].unique()).astype(float).min())) / 10,
                              key=str(y))
    select_param[y] = temp1

df_sear = (pd.read_csv("Searched_Records.csv")).drop('Unnamed: 0', axis=1)

try:
    for x in my_dict.keys():
        df_sear[x].iloc[0] = my_dict[x]
except NameError:
    print('')
for x in select_param.keys():
    df_sear[x].iloc[0] = select_param[x]

st.subheader("Custom User~")
st.markdown("<h3></h3>", unsafe_allow_html=True)
try:
    st.table(df_sear)  # .transpose())
except:
    st.error("Search Value not entered yet!")

l = [df_sear.iloc[0:1], dfst.iloc[0:100]]
df3 = pd.concat(l)


def convert_data(df_churn):
    df_churn = df_churn.reset_index(drop=True)
    empty_cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                  'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                  'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

    for i in empty_cols:
        df_churn[i] = df_churn[i].replace(" ", np.nan)

    df_churn.drop(['customerID', 'cluster_number'], axis=1, inplace=True)
    df_churn = df_churn.dropna()
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

    for i in binary_cols:
        df_churn[i] = df_churn[i].replace({"Yes": 1, "No": 0})

    # Encoding column 'gender'
    df_churn['gender'] = df_churn['gender'].replace({"Male": 1, "Female": 0})
    df_churn['Churn'] = df_churn['Churn'].replace({"Yes": 1, "No": 0})
    df_churn['PaymentMethod'] = df_churn['PaymentMethod'].replace({"Yes": 1, "No": 0})

    category_cols = ['PaymentMethod', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']

    # le = LabelEncoder()
    # for x in category_cols:
    #     df_churn[x] = le.fit_transform(df_churn[x])
    # enc = OneHotEncoder(handle_unknown='ignore')
    # df_churn[category_cols] = enc.fit_transform(df_churn[category_cols]).toarray()

    for cc in category_cols:
        dummies = pd.get_dummies(df_churn[cc], drop_first=False)
        dummies = dummies.add_prefix("{}#".format(cc))
        df_churn.drop(cc, axis=1, inplace=True)
        df_churn = df_churn.join(dummies)

    return df_churn


df4 = convert_data(df3)
df4['Unnamed: 0'] = [8999 + x for x in range(len(df4.index))]
l = list(df4.columns)
l = [l[-1]] + l[:-1]
df4 = df4[l]
to_pred = df4.iloc[0:1]

X1 = to_pred.loc[:, df4.columns != 'Churn']
y = to_pred.loc[:, df4.columns == 'Churn']
y_pred_prob = rfc_best.predict_proba(X1)
y_pred = rfc_best.predict(X1)
# st.info(('Churn Result: '+str(y_pred[0])))
st.markdown("<h3></h3>", unsafe_allow_html=True)
# st.warning(('Not Churn Probability: '+str(y_pred_prob[0,0])))
st.markdown("<h3></h3>", unsafe_allow_html=True)
# st.success(('Churn Probability: '+str(y_pred_prob[0,1])))

s = "Not Churned"
if (y_pred_prob[0, 0] < y_pred_prob[0, 1]):
    s = 'Churned'

d_nc = y_pred_prob[0, 0] * 360
d_c = y_pred_prob[0, 1] * 360


def make_pie(sizes, text, colors, labels):
    col = [[i / 255. for i in c] for c in colors]
    fig, ax = plt.subplots()
    ax.axis('equal')
    width = 0.45
    kwargs = dict(colors=col, startangle=180)
    outside, _ = ax.pie(sizes, radius=1, pctdistance=1 - width / 2, labels=labels, **kwargs)
    plt.setp(outside, width=width, edgecolor='white')
    kwargs = dict(size=15, fontweight='bold', va='center')
    ax.text(0, 0, text, ha='center', **kwargs)
    ax.set_facecolor('#e6eaf1')


c1 = (226, 33, 7)
c2 = (20, 20, 240)

make_pie([d_nc, d_c], s, [c2, c1], ['Probability(Not Churn): \n{0:.2f}%'.format(y_pred_prob[0, 0] * 100),
                                    'Probability (Churn): \n{0:.2f}%'.format(y_pred_prob[0, 1] * 100)])

# display code
st.pyplot()
st.header('Data Table~')
table1 = st.empty()
table1.dataframe(df8)

