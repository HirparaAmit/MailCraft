import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from conversation import get_default_conv_template
from flask import Flask, request, redirect, render_template

app = Flask(__name__)

def generate(style, topic, words='500', sender='Sender_Name', recipient='Recipient_Name'):
    tokenizer = AutoTokenizer.from_pretrained("GeneZC/MiniChat-3B", use_fast=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def get_model(device):
        if device == 'cuda':
            return AutoModelForCausalLM.from_pretrained("GeneZC/MiniChat-3B", use_cache=True, device_map="auto", torch_dtype=torch.float16).eval()
        else:
            return AutoModelForCausalLM.from_pretrained("GeneZC/MiniChat-3B", use_cache=True, device_map="cpu", torch_dtype=torch.float32).eval()

    model = get_model(device)
    conv = get_default_conv_template("minichat")

    question = f"""
        Generate an email with the following specifications and make sure to highlight the subject, according to the topic of the mail, in the very first line of the response:
      - Style: {style}
      - Word Limit: {words}
      - Topic: {topic}
      - Sender Name: {sender}
      - Recipient Name: {recipient}
    """

    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).to(device),
        do_sample=True,
        temperature=0.7,
        max_new_tokens=int(words),
    )
    output_ids = output_ids[0][len(input_ids[0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return output
    

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'GET':
        display = 'none'
        return render_template('index.html', display=display)
    else:
        display = 'block'
        style = request.form.get('style')
        words = request.form.get('words')
        topic = request.form.get('topic')
        sender = request.form.get('sender')
        recipient = request.form.get('recipient')
        response = generate(style, topic, words, sender, recipient)
        # response = "Lorem ipsum dolor sit amet consectetur adipisicing elit. Adipisci odio rem autem molestiae ut repellat accusantium impedit pariatur deserunt unde omnis facilis quaerat, libero qui illum? Enim quas illum dicta."
        return render_template('index.html', response=response, display=display)

if __name__ == '__main__':
    app.run(debug=True)