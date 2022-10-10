from flask import Flask, request, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

set(stopwords.words('english'))

app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')

    # convert to lowercase
    text1 = request.form['text1'].lower()

    text_final = ''.join(c for c in text1 if not c.isdigit())

    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    sa = SentimentIntensityAnalyzer()
    sentiment_dict = sa.polarity_scores(text=processed_doc1)
    print(sentiment_dict['compound'])
    compound = round((1 + sentiment_dict['compound']) / 2, 2)

    text = text_final
    positive_score = sentiment_dict['pos']
    negative_score = sentiment_dict['neg']
    neutral_score = sentiment_dict['neu']

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        sentiment_status = "Positive"
        print("Positive")

    elif sentiment_dict['compound'] <= - 0.05:
        sentiment_status = "Negative"
        print("Negative")

    else:
        sentiment_status = "Neutral"
        print("Neutral")

    send_to_frontend = [text,positive_score,negative_score,neutral_score,sentiment_status]

    return render_template('form.html', final=compound, text1=text_final, text2=sentiment_dict['pos'],
                           text5=sentiment_dict['neg'],
                           text4=compound, text3=sentiment_dict['neu'])


if __name__ == "__main__":
    app.run(debug=False)