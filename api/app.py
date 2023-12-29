import base64
import datetime
import json
import mimetypes
import os
from io import BytesIO

import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request
from keras import models
from keras.models import Model, model_from_json
from PIL import Image
from resUnit import ResUnit

app = Flask(__name__, static_url_path='/assets')

# model = models.load_model('model.keras', custom_objects={"ResUnit":ResUnit})
json_config = open('model.json').read()
model:Model = model_from_json(json_config, custom_objects={'ResUnit': ResUnit})
load_status = model.load_weights("model_weights.hdf5")
f = open('classes.json')
class_names = json.load(f)
f.close()

load_dotenv()
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

mimetypes.add_type("application/javascript", ".js")

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/img', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        request_body = request.json
        img = request_body['img']

        decoded_data = base64.b64decode(img.split(',')[1])
        new_img = Image.open(BytesIO(decoded_data))
        byte_arr = BytesIO()
        new_img.save(byte_arr, format='JPEG')
        byte_arr = byte_arr.getvalue()

        img_np = np.array(new_img)
        prediction = model.predict(img_np.reshape(1, *img_np.shape))
        prediction = prediction[0]
        max_index = prediction.argmax()
        max_value = prediction[max_index]
        content = f'{class_names[str(max_index)]} ({max_value*100:.2f}%)'
        now = datetime.datetime.now()
        now_str = now.strftime("%Y%m%d%H%M%S")

        data = {
            "username": "抵抗値判別",
            "content": content,
            # "avatar_url": ,""
        }
        body = {
            "payload_json": json.dumps(data)
        }
        files = {
            "attachment": (f'{now_str}.jpg', byte_arr, 'image/jpeg')
        }
        r = requests.post(
            WEBHOOK_URL,
            body,
            files=files
        )

        return r.content
    return {"status": "ok"}



