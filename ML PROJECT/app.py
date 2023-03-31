from flask import Flask, request, render_template
import pickle
import numpy as np
app = Flask(__name__)
# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from the form
    features=[float(x) for x in request.form.values()]
    x=[np.array(features)]

    # Use the model to make a prediction
    prediction = model.predict(x)[0]
    
    return render_template('index.html', prediction_text='Predicted Energy Usage: {:.2f} kWh'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)