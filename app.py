from flask import Flask, render_template, request, abort, send_file, redirect, jsonify, Response
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
import io
from werkzeug.utils import secure_filename
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

data = pd.read_csv('./data_set_new.csv',date_parser = True)
scaler = MinMaxScaler()
model_path = "./models"

# training parameter
lead_time = "0.5h"  # 0.5h, 1h, 3h, 6h
drop_out_rate = 0  # 0~1
seq_length = 24
train_size = int(len(data)*0.867078453393230)

# make data of training and test data
data_training = data[:train_size]
data_test = data[train_size-seq_length:]

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict",methods=["GET","POST"])
def predict():
    global scaler
    if request.method == 'POST':
        file = request.files['file']
        lead_time = request.values['leadtime']
        print(lead_time)
        if lead_time == "geumho_after_0.5h_drop-out_0_seq_length_24":
            column_lead_time = 8
        elif lead_time == "geumho_after_1h_drop-out_0_seq_length_24":
            column_lead_time = 9
        elif lead_time == "geumho_after_3h_drop-out_0_seq_length_24":
            column_lead_time = 10
        elif lead_time == "geumho_after_6h_drop-out_0_seq_length_24":
            column_lead_time = 11
        else:
            lead_time = "0.5h"
            column_lead_time = 8

        # make dataset of training
        data_training_drop = data_training.iloc[:,
                             [1, 2, 3, 4, 5, 6, 7, column_lead_time]]  # 8 : 0.5시간예측  9 : 1시간예측  10: 3시간 예측  11 : 6시간예측
        data_test_drop = data_test.iloc[:, [1, 2, 3, 4, 5, 6, 7, column_lead_time]]

        # data normalize of training data
        scaler.fit_transform(data_training_drop)

        # calculate predict Y
        model = tf.keras.models.load_model(os.path.join(model_path, lead_time))

        if file != '':
            filename = os.path.splitext(secure_filename(file.filename))[0]
            file_bytes = file.read()
            df = pd.read_csv(io.BytesIO(file_bytes),
                             infer_datetime_format=True, header=None)
            y_test = normalize_series(df)
            y_test = np.array(y_test).reshape(-1, 24, 7)
            y_pred = model.predict(y_test)
            dn = denormalize_series(y_pred)
            dn = dn.flatten().reshape(-1, 1)
            dn = pd.DataFrame(dn)
            output_stream = io.StringIO()
            dn.to_csv(output_stream)
            response = Response(
                output_stream.getvalue(),
                mimetype='text/csv',
                content_type='application/octet-stream',
            )
            filename = os.path.join(filename, "result")
            response.headers["Content-Disposition"] = "attachment; filename="+filename+".csv"
            return response
    else:
        return redirect("/")

def normalize_series(y_test):
    global scaler
    y_test = (y_test - scaler.data_min_[-1]) / scaler.data_range_[-1]
    return y_test

def denormalize_series(y_pred):
    global scaler
    y_pred = y_pred * scaler.data_range_[-1] + scaler.data_min_[-1]
    return y_pred
if __name__ == '__main__':
    app.run()
