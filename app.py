from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask"

@app.route('/data')
def return_data():
    data = {'values': [1, 2, 3, 4, 5, 6]}
    return jsonify(data)

if __name__ == '__main__':
    app.run(port=5000)
