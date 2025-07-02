from flask import Flask, request,render_template
import numpy as np
import pandas as pd

from src.pipelines.prediction_pipeline import PredictionPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score'))
        )
        test_df = data.get_data_as_data_frame()
        predcit_pipeline = PredictionPipeline(
            model_path='artifacts/model.pkl',
            preprocessor_path='artifacts/preprocessor.pkl'
        )
        results = predcit_pipeline.predict(test_df)
        print(results)
        return render_template('home.html', results=results[0])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    