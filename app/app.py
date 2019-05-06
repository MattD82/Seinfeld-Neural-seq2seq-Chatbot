from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort
from src.seq2seq_jerry_predict import JerryChatBot

app = Flask(__name__)

jerry_chat_bot = JerryChatBot()

jerry_chat_bot_convos = []

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return 'About Us'

@app.route('/jerry_reply', methods=['POST', 'GET'])
def jerry_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            jerry_chat_bot_convos.append('YOU: ' + sent)
            reply = jerry_chat_bot.reply(sent)
            jerry_chat_bot_convos.append('BOT: ' + reply)
    return render_template('jerry_reply.html', conversations=jerry_chat_bot_convos)


def main():
    jerry_chat_bot.test_run()
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    main()