import flask
from flask import send_file, request
from flask_cors import CORS, cross_origin
import json
import base64
import test
import os

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/process', methods=["GET", "POST"])
@cross_origin()
def inpaint():
    data = request.get_data()
    y = json.loads(data)
    image = y["image"][22:]
    mask = y["mask"][22:]
    
    imgdata = base64.b64decode(image)
    with open('input.png', 'wb') as f:
        f.write(imgdata)
            
    imgdata = base64.b64decode(mask)
    with open('mask.png', 'wb') as f:
        f.write(imgdata)
       
    test.inpaint("input.png", "mask.png", "output.png", "places_model/")
    
    return send_file("output.png", mimetype='image/PNG')
    
app.run(host= '0.0.0.0', port=11000)
