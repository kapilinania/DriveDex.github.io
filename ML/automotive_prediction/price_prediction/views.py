from django.shortcuts import render
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from django.template import loader
from django.http import HttpResponse
from django.shortcuts import render
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import OrderedDict
import math



model = pd.read_pickle('pikle2.pkl')
car_name=model["Car_Name"]
# print(car_name)
year=model["Year"]
distance=model["Distance"]
owner=model["Owner"]
fuel=model["Fuel"]
drive=model["Drive"]
label=model["CarLabel"]

# print(model)
# Create your views here.
def index(request):
    template = loader.get_template('index.html')
    return HttpResponse(template.render())


def car_list(request):
    car_data=model[['CarLabel','Car_Name','Year']]
    car_list=car_data.to_dict(orient='records')
    # print(car_list)
    return car_list



def bankfd(request):
    car_label=[car["CarLabel"] for car in car_list(request)]
    # print(car_label)
    car_name=[car["Car_Name"] for car in car_list(request)]
    year=[math.trunc(car["Year"]) for car in car_list(request)]
    combine=[]
    combine.append(car_label)
    combine.append(car_name)
    print(combine)
    data_dict=OrderedDict(zip(combine[0],combine[1]))


    context={
        "car_label":car_label,
        "car_name":car_name,   
        "year":year,  
        "data_dict":data_dict,    
    }

    return render(request, 'index.html',context)


def about(request):
    return render(request, 'about.html')


def contact(request):
    return render(request, 'contact.html')


# from
def form(request): 
    return render(request, 'form.html')


def getPredictions(q1, q2, q3, q4, q5, q6):
    
    # print(model)
    rf = model
   
    p_var = ['CarLabel', 'Year', 'Distance', 'Owner', 'FuelLabel', 'DriveLabel']
   
    X = model[p_var]
    y = model['Price']
    scaler_data = StandardScaler()
    fitx_data = scaler_data.fit_transform(X)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    rf = RandomForestRegressor(n_estimators=1000)
    # Training the Logistic Regression
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # r2_score(y_test, y_pred)  

    
    
    # input_features = ([[104  ,2021.0,     96339,      1 ,         1  ,         0]])
    input_features = [[q1, q2, q3, q4, q5, q6]]
    print(input_features)
    # print(input_features)
    new_data_reshaped = np.asarray(input_features)
    # standardize the input data

    scl_data = scaler_data.transform(new_data_reshaped)
    # print(scl_data)
    # Make predictions using the model
    prediction = rf.predict(scl_data)
    print(prediction)
    return str(prediction[0])

    # Convert the prediction to a string and return
    # if (prediction[0] == 0):
    #     return str('does not have any loan!')

    # #  return str('Term Deposit No')
    # else:
    #     return str('already have a loan!')
    # #  return str('Term Deposit yes')


# else:
#     return "Error"


def result(request):
    
    # Get values from the request
    q1 = int(request.GET.get('q1'))
    q2 = int(request.GET.get('q2'))
    q3 = int(request.GET.get('q3'))
    q4 = int(request.GET.get('q4'))
    q5 = float(request.GET.get('q5'))
    q6 = float(request.GET.get('q6'))
    


    
    
    # Call the getPredictions function
    result = getPredictions(q1, q2, q3, q4, q5, q6)
    # print(result)
    
    print(result)
    return render(request, 'index.html', {'result': result})
