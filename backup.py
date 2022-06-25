"""
Created on Thu Aug 26 23:57:33 2021

@author: TARUN
"""
import streamlit as st
import nsepy
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics 
from flask import Flask


    
st.title("Stock Price Predictor")
st.markdown("<hr/>",unsafe_allow_html=True)
sname=st.sidebar.text_input('Enter the stock symbol')
st.write (sname)
dur= st.sidebar.slider('No. of Days of historical data to be used', min_value=1, max_value=700, value=300, step=1)
algo_name=st.sidebar.selectbox("Select Algorithm",("Random Forest","Linear Regression","SVM","ARIMA"))


def gethdata(stockname,d):
    today=datetime.date.today()
    duration=d
    duration2=10
    start=today+datetime.timedelta(-duration)
    end1=today
    stockdata=nsepy.get_history(symbol=stockname,start=start,end=end1)
    return stockdata
def preprocess1(nd):
    v=nd.values
    v=np.append(v, [0])
    v=np.delete(v,0)
    return v
def randomforestreg():
    n_estimator= st.sidebar.slider("n_estimators",min_value=1,max_value=100,step=1)
    params[n_estimator]=n_estimator
    sv=st.selectbox('Pick The variable to be predicted', ["Prev Close","High","Last","Close","Deliverable Volume",'Volume','%Deliverble','VWAP','Low','Open','Trades','Turnover'])
    ndh=rf[[sv]]
    v=preprocess1(ndh)
    
    rf['ndhigh']=v#any selected variable not just high
    
    rf.reset_index(inplace = True)
    fs=st.multiselect("Select the features to be used as input",("Prev Close","High","Last","Close","Deliverable Volume",'Volume','%Deliverble','VWAP','Low','Open','Trades','Turnover'),['High'])
    X= rf[fs]   
    Y=rf['ndhigh']
   
    Xtestn=X.iloc[-1:]
    
    X.drop(index=X.index[-1], axis=0, inplace=True)
    Y.drop(index=Y.index[-1], axis=0, inplace=True)
    
    
    tst_sze=st.slider("Testing data size",min_value=0.1,max_value=1.0,step=0.1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = tst_sze, random_state=1234)
    regressor = RandomForestRegressor(n_estimators= params[n_estimator])
    regressor2 = RandomForestRegressor(n_estimators= params[n_estimator])
    regressor.fit(X_train,Y_train)
    
    test_data_prediction = regressor.predict(X_test)
    
    error_score = metrics.r2_score(Y_test, test_data_prediction)
    
    st.write("R squared error for ",sv," : ", error_score)
    
    
    # Plotting prediction_graph
    Y_test = list(Y_test)
    
    plt.subplots(figsize=(20, 10))   
    ax = plt.axes()
    ax.set_facecolor("#1b1e38")
    
    plt.plot(Y_test,'-o', color='blue', label = 'Actual '+sv+ 'Value')
    plt.plot(test_data_prediction,'-o', color='red', label='Predicted '+sv+' Value')
    
    plt.title('Actual Price vs Predicted Price')
    plt.xlabel('Number of values')
    
    
    plt.ylabel('Predicted and Actual '+sv)
    ax.grid()
    plt.legend()
    st.pyplot()
    test_data_predictionmon = regressor.predict(Xtestn)
    
   
    st.write("Tomorrow's PREDICTED ",sv," : ",test_data_predictionmon[0])
    
    
#end of random forest function


def linearreg():
    today=datetime.date.today()    
    duration2= st.sidebar.slider('No. of Days of testing data to be used:', min_value=1, max_value=dur, value=50, step=1)
    start=today+datetime.timedelta(-dur)
    end1=today+datetime.timedelta(-duration2)
    lstockdata=nsepy.get_history(symbol=sname,start=start,end=end1)  
    lrf=lstockdata
    ndh=lrf[['High']]
    v=preprocess1(ndh)
    ndh2=lrf[['Low']]
    v2=preprocess1(ndh2)
    lrf['ndhigh']=v
    lrf['ndlow']=v2 
    lrf.reset_index(inplace = True)
    fs=st.multiselect("Select the features to be used as input",("Prev Close","High","Last","Close","Deliverable Volume",'Volume','%Deliverble','VWAP','Low','Open','Trades','Turnover'),['High'])
    X= lrf[fs]   
    Y=lrf['ndhigh']
    X2= lrf[fs]  
    Y2=lrf['ndlow']
    Xtestn=X.iloc[-1:]
    Xtestnl=X2.iloc[-1:]
    X.drop(index=X.index[-1], axis=0, inplace=True)
    Y.drop(index=Y.index[-1], axis=0, inplace=True)
    X2.drop(index=X2.index[-1], axis=0, inplace=True)
    Y2.drop(index=Y2.index[-1], axis=0, inplace=True)
    X_train=X
    Y_train=Y
    X2_train=X2
    Y2_train=Y2    
    lr = LinearRegression()
    z=lr.fit(X_train,Y_train)
    lr2 = LinearRegression()
    z2=lr2.fit(X2_train,Y2_train)
    today=datetime.date.today()
       
    start=today+datetime.timedelta(-duration2)    
    stockdatalt=nsepy.get_history(symbol=sname,start=start,end=today)
    df1=stockdatalt
    test=df1[fs]    
    ndhi=df1[['High']]
    v11=preprocess1(ndhi)
    ndhi2=df1[['Low']]
    v22=preprocess1(ndhi2)
    test['ND high']=v11
    test['ND LOW']=v22
    df1.reset_index(inplace = True)
    testxh= df1[fs]
    lphigh=z.predict(testxh)
    testxl= df1[fs]
    lplow=z2.predict(testxl)
    Xtestn=X.iloc[-1:] 
    test['high prediction']=lphigh
    test['low prediction']=lplow
    #prediction df
    Xtesthl=test.iloc[-1:] 
    test.drop(index=test.index[-1],axis=0, inplace=True)
    error_score = metrics.r2_score(test['ND high'],  test['high prediction'])
    error_score2 = metrics.r2_score( test['ND LOW'], test['low prediction'])
    st.write("R squared error for HIGH : ", error_score)
    st.write("R squared error for LOW : ", error_score2)
    # plotting data
    a=[]
    i=0
    t=test.index.values
    st.write(test)
    for dt in t:
        a.append(dt.strftime("%m/%d/%Y"))
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = plt.axes()
    ax.set_facecolor("#1b1e38")
    xd = [datetime.datetime.strptime(d,"%m/%d/%Y").date() for d in a]
    ax.plot(xd, test['ND high'],'-o',label = "Actual next day High", color = 'red')
    ax.plot(xd, test['high prediction'],'-o', label = "predicted high by the model", color = 'orange')
    ax.plot(xd, test['ND LOW'],'-o',label = "Actual next day low", color = 'yellow')
    ax.plot(xd, test['low prediction'],'-o', label = "predicted low by the model", color = 'blue')
    plt.title('Actual Price vs Predicted Price')
    ax.grid()
    plt.legend()
    st.pyplot()
    st.write("Tomorrow's PREDICTED HIGH : ",Xtesthl['high prediction'].reset_index(drop=True).to_string(index=False))
    st.write("Tomorrow's PREDICTED LOW : ",Xtesthl['low prediction'].reset_index(drop=True).to_string(index=False))
 #End of linear regression  
 
rf=gethdata(sname,dur)
st.write(rf)
st.set_option('deprecation.showPyplotGlobalUse', False)
gv=st.multiselect("Select the feature to be represented in graph",("Prev Close","High","Deliverable Volume",'VWAP','Low','Trades','Open','Trades','Turnover'),["High"])
if not gv:
        st.error("Please select at least one feature.")
a=[]
i=0
t=rf.index.values
for dt in t:
    a.append(dt.strftime("%m/%d/%Y"))
fig, ax = plt.subplots(figsize=(40, 20))
xd = [datetime.datetime.strptime(d,"%m/%d/%Y").date() for d in a]
ax = plt.axes()
ax.set_facecolor("#1b1e38")
ax.plot(xd, rf[gv],'-o',label=gv)
plt.xlabel('Dates')
ax.grid()
plt.legend()
st.pyplot()

params=dict()
  
       
     



# PREDICTION
if algo_name=="Random Forest":
    randomforestreg()    

elif algo_name=="Linear Regression":
    linearreg()
    
else:
    st.error("ALGORITHM not added")




