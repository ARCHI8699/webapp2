# Important Modules
from flask import Flask, render_template, url_for, flash, redirect
import joblib
from flask import request
import numpy as np
import tensorflow
import os
from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user

app = Flask(__name__, template_folder='template')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.db'
app.config['SECRET_KEY'] = '619619'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(200))
    email = db.Column(db.String(200))
    password = db.Column(db.String(200))

    def is_active():
        return True


@login_manager.user_loader
def get(id):
    return User.query.get(id)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/home")
def homenew():
    return render_template("home2.html")


@app.route("/homeguest")
def homeguest():
    return render_template("home_guest.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/Malaria")
def Malaria():
    return render_template("index_mal.html")


@app.route("/Pneumonia")
def Pneumonia():
    return render_template("index_pneu.html")


@app.route("/MRI")
def MRI():
    return render_template("index_mri.html")


@app.route("/contact")
def Contact():
    return render_template("contact.html")


@app.route('/login', methods=['GET'])
def get_login():
    return render_template('login.html')


@app.route('/signup', methods=['GET'])
def get_signup():
    return render_template('signup.html')


@app.route('/login', methods=['POST'])
def login_post():
    email = request.form['email']
    password = request.form['password']
    user = User.query.filter_by(email=email).first()
    login_user(user)
    # return redirect('/')
    return render_template('home2.html')


@app.route('/signup', methods=['POST'])
def signup_post():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    user = User(username=username, email=email, password=password)
    db.session.add(user)
    db.session.commit()
    user = User.query.filter_by(email=email).first()
    login_user(user)
    # return redirect('/home')
    return render_template('home2.html')


@app.route('/logout', methods=['GET'])
def logout():
    logout_user()
    return redirect('/login')


dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

model_mal = load_model('model111.h5')
model_pneu = load_model("my_model.h5")
model_mri = load_model('tumor_prediction.h5')

# FOR THE FIRST MODEL

# call model to predict an image


def api(full_path):

    data = image.load_img(full_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    predicted = model_mal.predict(data)
    return predicted

# FOR THE SECOND MODEL


def api1(full_path):
    data = image.load_img(full_path, target_size=(64, 64, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255
    predicted = model_pneu.predict(data)
    return predicted

# FOR THE Third MODEL


def api2(full_path):
    data = image.load_img(full_path, target_size=(224, 224, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255
    predicted = model_mri.predict(data)
    return predicted


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST', 'GET'])
def upload_file():

    if request.method == 'GET':

        return render_template('index_mal.html')
    else:

        try:

            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'PARASITIC', 1: 'Uninfected',
                       2: 'Invasive carcinomar', 3: 'Normal'}
            result = api(full_name)
            print(result)

            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('predict_mal.html', image_file_name=file.filename, label=label, accuracy=accuracy)
        except:

            flash("Invalid selection!!", "danger")
            return redirect(url_for("Malaria"))


@app.route('/upload11', methods=['POST', 'GET'])
def upload11_file():

    if request.method == 'GET':

        return render_template('index_pneu.html')
    else:

        try:

            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'Normal', 1: 'Pneumonia'}
            result = api1(full_name)

            if (result > 50):
                label = indices[1]
                accuracy = result
            else:
                label = indices[0]
                accuracy = 100-result
            return render_template('predict_pneu.html', image_file_name=file.filename, label=label, accuracy=accuracy)
        except:
            flash("Invalid selection !!", "danger")
            return redirect(url_for("Pneumonia"))


@app.route('/upload111', methods=['POST', 'GET'])
def upload111_file():

    if request.method == 'GET':

        return render_template('index_mri.html')
    else:

        try:

            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'Tumorous', 1: 'Not Tumorous'}
            result = api2(full_name)
            print(result)

            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('predict_mri.html', image_file_name=file.filename, label=label, accuracy=accuracy)
        except:
            flash("Invalid selection !!", "danger")
            return redirect(url_for("MRI"))


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
