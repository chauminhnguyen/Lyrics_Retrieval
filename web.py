from flask import Flask, redirect, url_for, render_template, request
import utils.lyrics_retrieval as lr

app = Flask(__name__)

results_temp = [None]


@app.route("/lyric/<post_id>", methods=["POST", "GET"])
def find_lyric(post_id):
    global results_temp
    if request.method == "POST":
        query = request.form["inp"]
        results = lr.main(query)
        results_temp = results
        return render_template("lyric.html", results=results, index=0)
    else:
        return render_template("lyric.html", results=results_temp, index=int(post_id))


@app.route("/melody/<post_id>", methods=["POST", "GET"])
def func(post_id):
    return render_template("melody.html")


@app.route("/", methods=["POST", "GET"])
def home():
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
