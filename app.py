import os
import shutil
from uuid import uuid4
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
classes = ['apple','avocado','banana','dragon fruit','mango','mangosteen','orange','pineapple','rose apple','watermelon'] 
# classes = ['apple', 'banana','star fruit']
@app.route("/")
def index():
    return render_template("index.html")
# upload img to detec
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        new_model = load_model('D:\python\DACN2/modelCNN.h5')

        new_model.summary()
        test_image = tf.keras.utils.load_img('images\\'+filename,target_size=(180,198))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = new_model.predict(test_image)
        result1 = result[0]
        for i in range(10):
            if result1[i] == 1:
                break;
        prediction = classes[i]
        saved = os.path.join(APP_ROOT, 'classified/')
        label_folder = os.path.join(saved, prediction)
        os.makedirs(label_folder, exist_ok=True)
        new_destination = os.path.join(label_folder, filename)
        shutil.copy(destination, new_destination)

    return render_template("template.html",image_name=filename, text=prediction)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(debug=False)

