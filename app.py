from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model/model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    float_features=[float(x) for x in request.form.values()]
    final=[np.array(float_features)]
    print(float_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>=str(0.5):
        return render_template('index.html',pred='There are more chances of Malaria Outbreak\nProbability of Malaria Breeds occuring is {}'.format(output),
                               inp='Predicted for inputs:\nMax Temp: %.2f, Min Temp: %.2f, Humidity: %.1f Units(degree Celcius,percentage)'%(float_features[0],float_features[1],float_features[2]))
    else:
        return render_template('index.html',pred='Less chance of Malaria Outbreak\n Probability of Malaria Breeds occuring is {}'.format(output),
                               inp='Predicted for inputs:\nMax Temp: %.2f, Min Temp: %.2f, Humidity: %.1f  Units(degree Celcius,percentage)'%(float_features[0],float_features[1],float_features[2]))


if __name__ == '__main__':
    app.run(debug=True)
