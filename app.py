from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load all NLP pipelines
sentiment = pipeline("sentiment-analysis")
generator = pipeline("text-generation", model="gpt2")
unmask = pipeline("fill-mask")
qa = pipeline("question-answering")
summarizer = pipeline("summarization")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        task = request.form.get("task")
        text = request.form.get("input_text")

        try:
            if task == "Sentiment Analysis":
                result = sentiment(text)
            elif task == "Text Generation":
                result = generator(text, max_new_tokens=100)
            elif task == "Mask Fill":
                result = unmask(text)
            elif task == "Summarization":
                result = summarizer(text, max_length=150, min_length=40)
            elif task == "Question Answering":
                context = request.form.get("context")
                result = [qa(question=text, context=context)]
        except Exception as e:
            result = [{"Error": str(e)}]

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
