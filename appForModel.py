import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Poly_LR_Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #using below - parameters will be stored in list
    int_features = [int(x) for x in request.form.values()]
    #below is to convert single parameter to integer. DO not use if multiple parameters
    res = sum(d * 10**i for i, d in enumerate(int_features[::-1]))
    input = np.array([1])
    featsqr = res*res
    featcube = res*res*res
    input = np.append(input,res)
    input = np.append(input,featsqr)
    input = np.append(input,featcube)
    #convert 1D array to 2D array.
    input = input.reshape(1,-1)
    prediction = model.predict(input)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Salary at this level should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)