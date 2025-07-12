from transformers import pipeline

sentiment = pipeline("sentiment-analysis")
qa = pipeline("question-answering")
unmask = pipeline("fill-mask")
summarizer = pipeline("summarization")
translator = pipeline("translation", model="google-t5/t5-small")
