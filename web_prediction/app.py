import numpy as np
from flask import Flask, request, render_template
import pickle
from preprocess_function import preprocessing

app = Flask(__name__)

# Load the model
model = pickle.load(open('saved_model/model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.form['text']
    data1 = preprocessing(data)
    prediction = model.predict([data1])
    output = prediction[0]
	
    if output == 0:
        return render_template('index.html', prediction_text='TÍCH CỰC')

    return render_template('index.html', prediction_text='TIÊU CỰC')

if __name__ == '__main__':
    try:
        app.run()
    except:
        print("Error")