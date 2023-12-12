import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, render_template, request

app = Flask(__name__)

data = pd.read_csv('dataset/heart.csv')
X = data.drop('output', axis=1)
y = data['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trtbps = int(request.form['trtbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalachh = int(request.form['thalachh'])
        exng = int(request.form['exng'])
        oldpeak = float(request.form['oldpeak'])
        slp = int(request.form['slp'])
        caa = int(request.form['caa']) 
        thall = int(request.form['thall'])

        data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trtbps': [trtbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalachh': [thalachh],
            'exng': [exng],
            'oldpeak': [oldpeak],
            'slp': [slp],
            'caa': [caa],
            'thall': [thall]
        })

        prediction = model.predict(data)
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
