from flask import Flask, render_template, request
import pickle
import numpy as np

fileo = open('list.txt', 'rb')

model = pickle.load(fileo)

arr1 = np.array([[54350, 1, 8, 1234, 1876, 94, 11089]])

print(model.predict(arr1))

fileo.close()

app = Flask(__name__)


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data1 = request.form['ntransaction']
        data2 = request.form['fee']
        data3 = request.form['BST']
        data4 = request.form['GP']
        data5 = request.form['CI']
        data6 = request.form['OP']
        data7 = request.form['M2']

        data1 = float(data1)
        data2 = float(data2)
        data3 = float(data3)
        data4 = float(data4)
        data5 = float(data5)
        data6 = float(data6)
        data7 = float(data7)
        arr = np.array([[data1, data2, data3, data4, data5, data6, data7]])
        pred = model.predict(arr)
        pred_val = pred[0]
        str_value = str(pred_val)
        return render_template('prediction.html', bitcoin=str_value)

    return render_template('prediction.html')


if __name__ == "__main__":
    app.run(debug=True)
