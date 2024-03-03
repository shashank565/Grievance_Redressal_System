from flask import Flask, render_template, request, redirect, url_for
from flask_debug import Debug

import numpy as np
import pickle
import pandas as pd
import os
import re
import ftfy
import nltk
import smtplib
import numpy as np
import pandas as pd
import pickle as pkl
import pickle
from pathlib import Path
from nltk import PorterStemmer
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import smtplib
from keras import backend as K

app = Flask(__name__)

input_dir = 'input'
model_dir = 'model'


cword_file = 'cword_dict.pkl'
tokenizer_file = 'tokens.pkl'


class_names = ['Robbery', 'Murder', 'Assault', 'cyber _crime', 'Accident_case']

cList = pkl.load(open(os.path.join(input_dir,cword_file),'rb'))

trained_tokenizer = pkl.load(open(os.path.join(model_dir,tokenizer_file),'rb'))

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def load_trained_model(model_str_path, model_wt_path):
    f = Path(model_str_path)
    model_structure = f.read_text()
    trained_model = model_from_json(model_structure)
    trained_model.load_weights(model_wt_path)
    return trained_model

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

def clean_tweets(reviews):
    cleaned_review = []
    for review in reviews:
        review = str(review)
        review = review.lower()
    # if re.match("(\w+:\/\/\S+)", review) == None and len(review) > 10:
        review = ' '.join(re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", review).split())
        review = ftfy.fix_text(review)
        review = expandContractions(review)
        review = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", review).split())
        stop_words = set(stopwords.words('english'))
        word_tokens = nltk.word_tokenize(review) 
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        review = ' '.join(filtered_sentence)
        review = PorterStemmer().stem(review)
        cleaned_review.append(review)
    return cleaned_review


def res(texts):
    K.clear_session()
    input_dir = 'input'
    model_dir = 'model'

    cword_file = 'cword_dict.pkl'
    tokenizer_file = 'tokens.pkl'
    model_str_file = 'model_structure.json'
    model_weights_file = 'model_weights.h5'

    str_path = os.path.join(model_dir,model_str_file)
    wt_path = os.path.join(model_dir,model_weights_file)

    model = load_trained_model(str_path, wt_path)

    input_string = texts

    print(input_string)

    # test_data  = input_string.split(",")

    cleaned_text = clean_tweets(texts)
    print(cleaned_text)

    sequences_text_token = trained_tokenizer.texts_to_sequences(cleaned_text)
    data = pad_sequences(sequences_text_token, maxlen=250)

    

    result = model.predict(data)
    print(result)
    Y_pred = int(np.argmax(result))
    resl = class_names[Y_pred]
    K.clear_session()
    return resl

def mail(msg,rec_email):
    sender_email = "shashankyuva12@gmail.com"
    rec_email = rec_email
    password ="gneo ocpk revc kqni"
    message = msg

    server = smtplib.SMTP('smtp.gmail.com', 587)

    server.starttls()
    server.login(sender_email, password)
    print('login success')
    server.sendmail(sender_email, rec_email, message)
    print("Email has been sent to ", rec_email)

@app.route("/")
def index():
    return render_template("index1.html")

@app.route("/get", methods= ['POST'])
def get():
    return render_template("index.html")

@app.route("/admin", methods= ['POST'])
def admin():
    return render_template("admin.html")

@app.route("/views")
def views():
    df = pd.read_excel('res.xlsx')
    df1 = df[['Name','Date','Number','Email','Grivance']]
    df1 = df1.values
    res = list(df1)
    return render_template("view.html", resl = res)

@app.route("/act", methods= ['POST'])
def act():
    val = []
    val_j = []
    if request.method == "POST":
        val.append(int(request.form["q1"]))
        val_j.append(request.form["j"])
        if val_j[0] == "Accept":
            df_1 = pd.read_excel('accept.xlsx')
            df = pd.read_excel('res.xlsx')
            print(val)
            print(df.columns)
            indexNames = df[ df['Number'] == val[0] ].index
            ui = df[ df['Number'] == val[0] ].values
            ui = list(ui[0])
            print(ui)
            df_1.loc[len(df)] = ui
            df_1.to_excel('accept.xlsx',index=False)
            df.drop(indexNames , inplace=True)
            df.to_excel('res.xlsx',index=False)

            msg = 'We are writing to confirm that your recent grievance has been received and processed through our system. We acknowledge and validate your concerns regarding the matters you have brought to our attention. We are pleased to inform you that your complaint has been duly documented.'
            rec_email = ui[5]
            rec_email = str(rec_email)
            print(rec_email)
            mail(msg,rec_email)
            return render_template("accept.html")
        
        else:
            df_2 = pd.read_excel('reject.xlsx')
            df = pd.read_excel('res.xlsx')
            print(val)
            indexNames = df[ df['Number'] == val[0] ].index
            ui = df[ df['Number'] == val[0] ].values
            df_2.loc[len(df)] = ui[0]
            ui = list(ui[0])
            print(ui)
            df_2.to_excel('reject.xlsx',index=False)
            df.drop(indexNames , inplace=True)
            df.to_excel('res.xlsx',index=False)

            msg = 'Your complaint has been rejected contact the nearby police for further info.'
            rec_email = ui[5]
            rec_email = str(rec_email)
            print(rec_email)
            # mail(msg,rec_email)
            return render_template("reject.html")
    return redirect(url_for("views"))


@app.route("/login",methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]
        r1 = pd.read_excel('user.xlsx')

        for index, row in r1.iterrows(): 
            if row["u_name"]==str(uname) and row["pass"]==str(passw):
                df = pd.read_excel('res.xlsx')
                df1 = df[['Name','Date','Number','Email','Grivance']]
                df1 = df1.values
                res = list(df1)
                return render_template("view.html", resl = res)
        return render_template("login.html")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        # mail = request.form['mail']
        passw = request.form['passw']

        r1 = pd.read_excel('user.xlsx')
        new_row = {'u_name':uname,'pass':passw}
        r1 = r1.append(new_row, ignore_index=True)

        r1.to_excel('user.xlsx')
        
        print("Records created successfully")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route('/predict',methods = ['POST'])
def predict():
    ui = []
    inp = []
    if request.method == 'POST':
        ui.append(request.form['name'])
        ui.append(request.form['age'])
        ui.append(request.form['Gender'])
        ui.append(request.form['Area'])
        ui.append(request.form['phone'])
        ui.append(request.form['email'])
        ui.append(request.form['day'])
        ui.append(request.form['Grivance'])
        inp.append(request.form['Grivance'])
        print("its come to 1 if")
    
    resl = res(inp)
    ui.append(resl)
    print(ui)
    df = pd.read_excel('res.xlsx')
    # df = df.append(ui)
    df.loc[len(df)] = ui
    df.to_excel('res.xlsx',index=False)
    return render_template('result.html',u=ui)

if __name__ == "__main__":
    app.run(host="127.0.0.9", port=8080, debug=True)