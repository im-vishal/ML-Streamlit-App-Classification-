# importing liabraries:
from random import seed
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# from sklearn importing datasets:
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection

matplotlib.use('Agg')
from PIL import Image

# Set Title
st.title('Classification Machine Learning App')
image = (Image.open('download.png'))
st.image(image, use_column_width=True)


def main():
    activities = ['EDA', 'Visualisation', 'Model', 'About Me']
    option = st.sidebar.selectbox('Selection option:', activities)

    data = st.file_uploader("Upload dataset:", type='csv')
        
    if data is not None:
        st.success('Data Successfully loaded')
        df = pd.read_csv(data)
        st.dataframe(df.head(50))

# EDA Part
    if option=='EDA':
        st.subheader('Exploratory Data Analysis')

        if data is not None:

            if st.checkbox("Display shape"):
                st.write(df.shape)
            if st.checkbox("Display Columns"):
                st.write(df.columns)
            if st.checkbox("Select multiple columns"):
                selected_columns = st.multiselect("Select preferred columns:", df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)

            if st.checkbox("Display Summary"):
                st.write(df1.describe().T)

            if st.checkbox('Display Null Values'):
                st.write(df.isnull().sum())

            if st.checkbox('Display the data types'):
                st.write(df.dtypes)
            if st.checkbox('Display Correlation of various columns'):
                st.write(df.corr())






# Visualisation Part
    elif option == 'Visualisation':
        st.subheader("Visual Representation of Data")

               
        if data is None:
            st.warning('Upload the data first')
        else:
            
            if st.checkbox('Display columns'):
                st.write(df.columns)
            if st.checkbox('Select multiple columns to plot'):
                selected_columns = st.multiselect('Select your preferred columns', df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)

            st.set_option('deprecation.showPyplotGlobalUse', False)
            
            if st.checkbox('Display Heatmap'):
                fig, ax = plt.subplots()
                st.write(sns.heatmap(df1.corr(),vmax=1,square=True,annot=True,fmt='.5f',cmap='viridis'))
                st.pyplot(fig)

            if st.checkbox('Display Pairplot'):
                st.write(sns.pairplot(df1,diag_kind='kde'))
                st.pyplot()

            if st.checkbox('Display Pie Chart'):
                all_columns = df.columns.to_list()
                pie_columns = st.selectbox('select column to display', all_columns)
                pie_chart = df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_chart)
                st.pyplot()

# Model Building Part
    elif option == 'Model':
        st.subheader("Model Building")

               
        if data is None:
            st.warning('Upload the data first')
        else:            
            if st.checkbox('Select Multiple Columns'):
                new_data = st.multiselect('Select your preferred columns. Note: Let your target variable be the last column to be selected',
                df.columns)
                df1 = df[new_data]
                st.dataframe(df1)

                # dividing data into X and y
                X = df1.iloc[:,0:-1]
                y = df1.iloc[:,-1]

                seed = st.sidebar.slider('Seed',1,200)
                classifier_name=st.sidebar.selectbox('Select your preferred Classifier:',('KNN','SVM','LR','naive_bayes','decision tree'))

                def add_parameter(name_of_clf):
                    param = dict()
                    if name_of_clf == 'SVM':
                        C = st.sidebar.slider('C', 0.01, 15.0)
                        param['C'] = C
                    if name_of_clf == 'KNN':
                        K = st.sidebar.slider('K', 1, 15)
                        param['K'] = K
                    return param

                # calling the function
                params = add_parameter(classifier_name)

                # defining a function for our classifier
                def get_classifier(name_of_clf, params):
                    clf = None
                    if name_of_clf == 'SVM':
                        clf = SVC(C = params['C'])
                    elif name_of_clf == 'KNN':
                        clf = KNeighborsClassifier(n_neighbors= params['K'])
                    elif name_of_clf == 'LR':
                        clf = LogisticRegression()
                    elif name_of_clf == 'naive_bayes':
                        clf = GaussianNB()
                    elif name_of_clf == 'decision tree':
                        clf = DecisionTreeClassifier()
                    else:
                        st.warning('Select your choice of algorithm')
                    return clf

                clf = get_classifier(classifier_name, params)

                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=seed)

                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                st.write('Predictions:', y_pred)
                accuracy = accuracy_score(y_test, y_pred)

                st.write('Name of Classifier:', classifier_name)
                st.write('Accuracy:', accuracy)

                
# About me Part
    else:
        st.write(' This is an interactive web page for our ML project, feel free to use it')
        st.markdown('*Created by:*  Vishal Pandey')
        st.balloons()



if __name__ == '__main__':
    main()
