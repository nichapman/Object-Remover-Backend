import flask
from flask import send_file, request
from flask_cors import CORS, cross_origin
import json
import base64
import inpaint
import os

IMAGE_DATA_PREFIX_LENGTH = 22;

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/process', methods=["GET", "POST"])
@cross_origin()
def inpaint():
    #load in json containing input images
    data = request.get_data()
    data_object = json.loads(data)
    
    #trim unnecessary prefix
    image = data_object["image"][IMAGE_DATA_PREFIX_LENGTH:]
    mask = data_object["mask"][IMAGE_DATA_PREFIX_LENGTH:]
    
    #save the image input locally
    image_data = base64.b64decode(image)
    with open('input.png', 'wb') as f:
        f.write(image_data)
            
    #save the mask input locally
    mask_data = base64.b64decode(mask)
    with open('mask.png', 'wb') as f:
        f.write(mask_data)
       
    #call inpainting script with input images and path to model, save to output.png
    inpaint.process("input.png", "mask.png", "output.png", "places_model/")
    
    #return processed output image
    return send_file("output.png", mimetype='image/PNG')
    
app.run(host= '0.0.0.0', port=11000)