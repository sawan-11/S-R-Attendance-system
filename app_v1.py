"""
This file is used for converting the model into an API using Flask.
"""
# Default Imports
from flask import Flask, jsonify, request, render_template, session, flash,redirect,url_for
import credentials as cred
import sys
from flask_cors import CORS
import pyodbc
from flask_mail import Mail
import json
# ms sql connection server and database name
SERVER = "LAPTOP-LVK4MC26\SAWAN"
DATABASE = "attendance"

with open('config.json', 'r') as c:
    params = json.load(c)["params"]

# %%
flag = 0
app = Flask(__name__)
app.secret_key = 'super-secret-key'
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT='465',
    MAIL_USE_SSL=True,
    MAIL_USERNAME=params['gmail_user'],
    MAIL_PASSWORD=params['gpass'])

# email sending functionality
mail = Mail(app)
CORS(app)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', params=params)


@app.route('/status', methods= ['GET'])
def status():
    try:
        global flag
        print("Processing the parameters")
        # print(request)
        output = cred.FaceRecognitionCredentials.loadConfigurations(request)
        print("loaded the parameters with status:", output)
        var = {"result": output}
        return jsonify(var)
    except:
        err = {"Error Identified": sys.exc_info()[1]}
        print(err)
    return redirect(url_for('status', EMP_ATTENDANCE_IDENTIFY_MSSQL_IP_ADDRESS='LAPTOP-LVK4MC26\SAWAN', EMP_ATTENDANCE_IDENTIFY_MSSQL_DATABASE='attendance', DATASET_FOLDER_NAME='datasets'))


@app.route('/identifyFacialImage')
def identify_facial_image():
    try:
        import face_recognize_v1 as fr
        fr.identifyFacialImage()
        print("getPatientRecord")
        var = {"Identify": "success"}
        return jsonify(var)
    except:
        err = {"Error Identified": sys.exc_info()[1]}
        print(err)


@app.route('/createFacialData', methods=['GET', 'POST'])
def record_facial_image():
    try:
        if request.method == 'POST':
            name = request.form['emp_name']
            empId = request.form['emp_id']
            print("Name is " + name + " and Employee ID is " + empId)
            import create_data_v1 as cd
            var = cd.recordFacialImage(name, empId)
            print("recordFacialImage")
            var["capture"] = "success"

            return jsonify(var), render_template('index.html', name=name, emp_Id = empId, params=params)
    except:
        err = {"Error Identified": sys.exc_info()[1]}
        print(err)
    return render_template('create_data.html', params=params)


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html', params=params)


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        phone = request.form.get('phone')
        email = request.form.get('email')
        msg = request.form.get('msg')
        if name:
            mssqlConnection = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + SERVER + ';DATABASE=' + DATABASE + '; Trusted_connection=yes;')
            entry = "INSERT INTO contact(name, email, phone, message, date) VALUES "" ('" + name + "', '" + str(email) + "', '" + phone + "', '" + msg + "',getdate());"
            print('Got the mssqlConnection:', mssqlConnection)
            with mssqlConnection:
                mssqlCursor = mssqlConnection.cursor()
                print('Executing the query:', entry)
                mssqlCursor.execute(entry)
                mssqlCursor.commit()
                flash("Your message has been send thank you for contacting us!", category="success")
                mail.send_message('New message of S.R Attendance from '+name,
                                  sender=email,
                                  recipients=[params['gmail_user']],
                                  body=msg + "\n" + phone)
            return render_template('contact.html', name=name, phone=phone, email=email, msg=msg, params=params)
        else:
            flash("Your message has not been send!", category="danger")
            return render_template('contact.html', params=params)
    return render_template('contact.html', params=params)

# for admin access
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    mssqlConnection = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + SERVER + ';DATABASE=' + DATABASE + '; Trusted_connection=yes;')
    query = "SELECT * FROM Attendance_record ORDER BY id ASC"
    if 'admin' in session and session['admin'] == params['username']:
        with mssqlConnection:
            mssqlCursor = mssqlConnection.cursor()
            print('Executing the query:', query)
            mssqlCursor.execute(query)
            data = mssqlCursor.fetchall()
            mssqlCursor.commit()
        return render_template('dashboard.html', server=SERVER, data=data, params=params)

    if request.method == 'POST':
        username = request.form.get('Username')
        password = request.form.get('pass')
        if username == params['username'] and password == params['upass']:
            session['admin'] = username
            print('Got the mssqlConnection:', mssqlConnection)
            with mssqlConnection:
                mssqlCursor = mssqlConnection.cursor()
                print('Executing the query:', query)
                mssqlCursor.execute(query)
                data = mssqlCursor.fetchall()
                mssqlCursor.commit()
            return render_template('dashboard.html', data=data, params=params)
    return render_template('admin.html')

@app.route('/ResetData')
def reset_data():
    mssqlConnection = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + SERVER + ';DATABASE=' + DATABASE + '; Trusted_connection=yes;')
    delete_query = "delete Top (1000) from Attendance_record "
    try:
        with mssqlConnection:
            mssqlCursor = mssqlConnection.cursor()
            print('Executing the query:', delete_query)
            mssqlCursor.execute(delete_query)
            mssqlCursor.commit()
            return render_template('dashboard.html',params=params)
    finally:
            return render_template('dashboard.html',params=params)


@app.route('/searchData', methods=['GET','POST'])
def search_data():
    try:
        if request.method == 'POST':
            id = request.form['emp_id']
            Emp_id = id
            mssqlConnection = pyodbc.connect(
                        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + SERVER + ';DATABASE=' + DATABASE + '; Trusted_connection=yes;')
            search_query ="(select Id,name,InTime,OutTime,InputPerDay from Attendance_record where Id = '"+Emp_id+"')"
            with mssqlConnection:
                mssqlCursor = mssqlConnection.cursor()
                print('Executing the query:', search_query)
                mssqlCursor.execute(search_query)
                data = mssqlCursor.fetchall()
                mssqlCursor.commit()
                return render_template('searched_employee.html', id=id, data=data, params=params)
    except:
        err = {"Error Identified": sys.exc_info()[1]}
        print(err)
    return render_template('filtered_employee.html',params=params)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
