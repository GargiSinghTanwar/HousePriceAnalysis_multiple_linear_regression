
import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


app = Flask(__name__)
model = pickle.load(open('house_price.pkl','rb')) 


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    exp = int(request.args.get('exp'))
    exp0 = int(request.args.get('exp0'))
    exp1 = int(request.args.get('exp1'))
    exp2 = int(request.args.get('exp2'))
    exp3 = int(request.args.get('exp3'))
    exp4 = int(request.args.get('exp4'))
    prediction = model.predict([[exp,exp0,exp1,exp2,exp3,exp4]])
    
        
    return render_template('index.html', prediction_text='Regression Model  has predicted price for given area is : {}'.format(prediction))
if __name__ == "__main__":
    app.run(debug=True)
