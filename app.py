from flask import Flask, render_template,request
import pickle as pk
import pandas as pd
from Heart_module import OrdinalEncoder,NRootTransformer


app=Flask(__name__)


with open('Heart_model.pkl', 'rb') as f:
    model = pk.load(f)


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    sex=(request.form.get('sex'))
    ageCategory=(request.form.get('ageCategory'))
    bmi=float(request.form.get('bmi'))
    genHealth=(request.form.get('genHealth'))
    physicalActivity=(request.form.get('physicalActivity'))
    physicalHealth=float(request.form.get('physicalHealth'))
    mentalHealth=float(request.form.get('mentalHealth'))
    sleepTime=float(request.form.get('sleepTime'))
    diffWalking=(request.form.get('diffWalking'))
    smoking=(request.form.get('smoking'))
    alcoholDrinking=(request.form.get('alcoholDrinking'))
    kidneyDisease=(request.form.get('kidneyDisease'))
    asthma=(request.form.get('asthma'))
    skinCancer=(request.form.get('skinCancer'))
    stroke=(request.form.get('stroke'))
    diabetic=(request.form.get('diabetic'))

    print(sex,ageCategory,bmi,genHealth,physicalActivity,physicalHealth,mentalHealth,sleepTime,diffWalking,smoking,alcoholDrinking,kidneyDisease,asthma,skinCancer,stroke,diabetic)
    
    data = {
        'Sex': [sex],
        'AgeCategory': [ageCategory],
        'BMI': [bmi],
        'GenHealth': [genHealth],
        'PhysicalActivity': [physicalActivity],
        'PhysicalHealth': [physicalHealth],
        'MentalHealth': [mentalHealth],
        'SleepTime': [sleepTime],
        'DiffWalking': [diffWalking],
        'Smoking': [smoking],
        'AlcoholDrinking': [alcoholDrinking],
        'KidneyDisease': [kidneyDisease],
        'Asthma': [asthma],
        'SkinCancer': [skinCancer],
        'Stroke': [stroke],
        'Diabetic': [diabetic]
    }

    df = pd.DataFrame(data)
    print(df)
    predict=model.predict(df)
    print(predict)
    predict=str(predict)
    return predict

if __name__=="__main__":
    app.run(debug=True)
