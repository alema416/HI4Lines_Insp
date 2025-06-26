from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
    # bind to 0.0.0.0 so Docker publishes it
    app.run(host="0.0.0.0", port=8004)
