from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from flask import Flask, request, redirect, render_template

app = Flask(__name__)

def generate(style, topic, words='500', sender='Sender_Name', recipient='Recipient_Name'):
    llm = CTransformers(model='./llama-2-7b-chat.ggmlv3.q8_0.bin',
                model_type='llama',
                config={'max_new_tokens': 1000,
                        'temperature': 0.01})

    template = """
        Generate an email with the following specifications and make sure to highlight the subject, according to the topic of the mail, in the very first line of the response:
      - Style: {style}
      - Word Limit: {words}
      - Topic: {topic}
      - Sender Name: {sender}
      - Recipient Name: {recipient}
  """

    prompt = PromptTemplate(input_variables=["style", "words", "topic", "sender", "recipient"], template=template,)
    response=llm(prompt.format(style=style, words=words, topic=topic, sender=sender, recipient=recipient))
    return response

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        style = request.form.get('style')
        words = request.form.get('words')
        topic = request.form.get('topic')
        sender = request.form.get('sender')
        recipient = request.form.get('recipient')
        response = generate(style, topic, words, sender, recipient)
        return render_template('index.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)