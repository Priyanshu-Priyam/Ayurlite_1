from flask import Flask, render_template, request, jsonify,url_for,flash,redirect,session
from werkzeug.utils import secure_filename
import pickle
from model import extract_features
from resources import get_bucket, get_buckets_list,read_bucket_objects
import numpy as np

""""
from model import extract_features
from model import DecisionTreeClassifier
from model import X_test
from model import X_train

"""




app = Flask(__name__)
app.secret_key = "secret"
classifier = pickle.load(open('model.pkl', 'rb'))

#app.config["UPLOAD_FOLDER"] = "static/"

@app.route('/')
def index():
    return render_template('index.html')


#@app.route('/display', methods = ['GET', 'POST'])


@app.route('/files')
def files():
    my_bucket = get_bucket()
    summaries = my_bucket.objects.all()
    flash('File uploaded successfully')
    return render_template('content.html', content=my_bucket)


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        my_bucket = get_bucket()
        my_bucket.Object(file.filename).put(Body=file)
        x_user = extract_features(read_bucket_objects(filename))
        prediction = classifier.predict(x_user)
        if prediction==0:
            con= "Your BP is in the range of Hypotension and You are Diabetic"
        elif prediction==1:
            con = "Your Bp is the range of Normal Blood Pressure"
        elif prediction==2:
            con =" Your Bp is the range of Elevated Blood Pressure"
        else:
            con = " Your Bp is the range of Hypertension are Non-Diabetic"
        
        content= con
    return render_template('content.html',content=content)


def save_file():
    
    if request.method == 'POST':
        f = request.files['file']
        #content =f.read()
        #str1 = content.decode('UTF-8')
        filename = secure_filename(f.filename)
        f.save(app.config['UPLOAD_FOLDER'] + filename)

        file = open(app.config['UPLOAD_FOLDER'] + filename,"r")
        content = file.read()


        import numpy as np
        #fn=input("Enter the name of the file")ß
        x_user = extract_features(app.config['UPLOAD_FOLDER']+filename)
        #<--------------- change the location of filename----------------->
        prediction = classifier.predict(x_user)

        content= prediction 


    return render_template('content.html', content=content) 



"""


 
"""
if __name__ == '__main__':
    app.run(port=5003, debug = True)


   