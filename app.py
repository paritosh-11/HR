from tkinter import HORIZONTAL
from turtle import title
from unittest import result
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np
from sklearn import preprocessing
import pickle as pickle
from st_aggrid import AgGrid
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs

model =  pickle.load(open('model.pkl','rb'))

#Prediciting
def predict(data):
    le = preprocessing.LabelEncoder()
    data['salary']=le.fit_transform(data['salary'])
    data['Departments']=le.fit_transform(data['Departments'])
    X=data[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'Departments', 'salary']]
    y_pred_pickle = model.predict(X)
    return y_pred_pickle

#sidebar
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=['Home','Prediction','Visualization'],
        icons=['house-door-fill','graph-up','table'],
        default_index=0
    )
if selected == 'Home':
    st.header("Employee Churn Prediction")
    st.subheader("Do you want to know whether your employees wish to leave your firm or not?")
    st.write("Employee churn prediction aims to find whether your valuable and productive employee(s) wishes to leave your organization or not.")
    st.write("Employees are the essential pillars of any working organization or corporate, their involvement and work are what make the company's development increase.")
    st.write("Further, it increases the company's reputation, performance among other companies, more fresh employees for recruitment, and the company's share in the stock market.")
    st.write("But if they quit jobs unexpectedly, it may incur a huge cost to the organization and affect the benefits mentioned above.")


if selected == 'Prediction' :
    with st.sidebar:
        predict_method = option_menu(
            menu_title=None,
            options=['Group','Single'],
            icons=['people-fill','person-fill'],
            default_index=0,
            orientation='horizontal',
            styles={
                "nav-link":{
                    "font-weight":"500"

                }
            }
    )
    if predict_method == 'Group':
        st.subheader("Group Prediction")
        data_file = st.file_uploader("Upload CSV",type=["csv"])
        if data_file is not None :
            #file_details = {"filename":data_file.name, "filetype":data_file.type,"filesize":data_file.size}
            #st.write(file_details)
            df = pd.read_csv(data_file)

            show_dat = st.checkbox('Show Data')
            if show_dat:
                st.dataframe(df)
            df_with_id_name = pd.DataFrame(df,columns=['id','name'])#dataframe with id and name,for merging with predicted dataframe
            

            if st.button("Predict"):
                predicted_df = predict(df)
                predicted_df = pd.DataFrame(predicted_df,columns=['left'])

                df_with_id = pd.DataFrame(df,columns=['id'])
                df_with_id_list = df_with_id.values.tolist()#2d array with id
                predicted_df_list = predicted_df.values.tolist()##2d array with prediction(left)

                
                predict_list = [] 
                id_list = []
                #for converting the 2d array to corresponding 1d array
                for i in range(len(df_with_id_list)):
                    for j in range(len(df_with_id_list[i])):
                        predict_list.append(predicted_df_list[i][j])
                        id_list.append(df_with_id_list[i][j])
                #converting 1 and 0 to Yes and No
                predict_list_with_yes_or_no= []
                for i in range(len(predict_list)):
                    if predict_list[i] == 1:
                        predict_list_with_yes_or_no.append("Yes")
                    else:
                        predict_list_with_yes_or_no.append("No")
                #dictionary ,later to converted to dataframe
                df_dictionary = {'id': id_list,
                         'left': predict_list_with_yes_or_no} 
                appended_df = pd.DataFrame.from_dict(df_dictionary)
                merged_df = pd.merge(df_with_id_name,appended_df )
                AgGrid(merged_df,fit_columns_on_grid_load=True,theme="streamlit")

    if predict_method == 'Single':
        st.subheader("Single Employee Prediction")
        with st.form(key='my_form'):
            name = st.text_input(label='Name')
            satisfaction_level =  st.number_input(label="Satisfaction Level", min_value=0.0, max_value=1.0,step=0.1)
            last_evaluation =  st.number_input(label="last_evaluation", min_value=0.0, max_value=1.0,step=0.1)
            number_project =  st.number_input(label="number_project",step=1)
            average_montly_hours = st.number_input(label="average_montly_hours",step=1)
            time_spend_company =  st.number_input(label="time_spend_company",step=1)
            work_accident =  st.selectbox(
                                    'work accedent',
                                    ('yes', 'no'))
            promotion_last_5years = st.selectbox(
                                    'promotion_last_5years',
                                    ('yes','no'))
            Departments = st.selectbox(
                                    'Department',
                                    ('sales', 'IT', 'hr','accounting','technical','support','management'))
            salary =  st.selectbox(
                                    'Salery',
                                    ('high', 'medium', 'low'))

            submit_button = st.form_submit_button(label='Predict')

            if submit_button:
                if work_accident == 'yes':
                    work_accident = 1
                if work_accident == 'no':
                    work_accident = 0
                if promotion_last_5years == 'yes':
                    promotion_last_5years = 1
                if promotion_last_5years == 'no':
                    promotion_last_5years = 0
                data = {
                            "satisfaction_level":satisfaction_level,
                            "last_evaluation":last_evaluation,
                            "number_project":number_project,
                            "average_montly_hours":average_montly_hours,
                            "time_spend_company":time_spend_company,
                            "Work_accident":work_accident,
                            "promotion_last_5years":promotion_last_5years,
                            "Departments":Departments,
                            "salary":salary
                        }
                print(data)
                df = pd.DataFrame([data])
                le = preprocessing.LabelEncoder()
                df['salary']=le.fit_transform(df['salary'])
                df['Departments']=le.fit_transform(df['Departments'])
                prediction = model.predict(df)
                if prediction[0] == 1:
                    st.error("**{}** will **leave** the company".format(name))
                if prediction[0] == 0:
                    st.success("**{}** will **stay** in the company".format(name))
                print(prediction)
                


        
if selected == 'Visualization':
    st.subheader("Visualization")
    data_file = st.file_uploader("Upload CSV",type=["csv"])
    if data_file is not None :

        uploaded_df = pd.read_csv(data_file)
        show_dat = st.checkbox('Show Data')
        if show_dat:
            st.dataframe(uploaded_df)
        result_df = predict(uploaded_df)
        result_df = pd.DataFrame(result_df,columns=['left'])
        df_with_id = pd.DataFrame(uploaded_df,columns=['id'])
        df_with_id_list = df_with_id.values.tolist()#2d array with id
        result_df_list = result_df.values.tolist()

        predict_list = [] 
        id_list = []
                #for converting the 2d array to corresponding 1d array
        for i in range(len(df_with_id_list)):
            for j in range(len(df_with_id_list[i])):
                predict_list.append(result_df_list[i][j])
                id_list.append(df_with_id_list[i][j])
                #converting 1 and 0 to Yes and No
        predict_list_with_yes_or_no= []
        for i in range(len(predict_list)):
            if predict_list[i] == 1:
                predict_list_with_yes_or_no.append("Yes")
            else:
                predict_list_with_yes_or_no.append("No")
                #dictionary ,later to converted to dataframe
        df_dictionary = {'id': id_list,
                         'left': predict_list_with_yes_or_no} 
        appended_df = pd.DataFrame.from_dict(df_dictionary)
        merged_df = pd.merge(uploaded_df,appended_df )
        
        def show_graph(option):
            fig = plt.figure(figsize=(10, 4))
            sns.countplot(x=option,
                hue = 'left',
                data = merged_df)
            plt.xticks(rotation=90)
            st.pyplot(fig)

        option = st.selectbox(
             'Select an attribute',
            ("satisfaction_level","last_evaluation",'number_project','average_montly_hours', 'time_spend_company', 'promotion_last_5years',"Departments","salary",'left'))
        if option:
            show_graph(option)




 