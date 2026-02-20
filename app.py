from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def encrypt_route():
    return render_template("index.html")


#
app.run(host='0.0.0.0', port=8080, debug=True)