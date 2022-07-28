from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model
from PIL import Image #use PIL
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def init():
    if request.method == 'POST':
        file = request.files['file']
        print("File Received")
        filename = secure_filename(file.filename)
        print(filename)
        # Open the image form working directory
        image = Image.open(file)
        model = load_model("Pneumonia")
        img = np.asarray(image)
        img.resize((150,150,3))
        img = np.asarray(img, dtype="float32") #need to transfer to np to reshape
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) #rgb to reshape to 1,100,100,3
        pred=model.predict(img)
        return(render_template("index.html", result=str(pred)))
    else:
        return(render_template("index.html", result="WAITING"))
if __name__ == "__main__":
    app.run()
