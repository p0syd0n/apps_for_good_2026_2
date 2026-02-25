from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def main():
    print("Main hit")
    return render_template("index.html")


#

app.run(host='0.0.0.0', port=8080, debug=True)