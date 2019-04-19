from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort
from src.seq2seq_jerry_predict import JerryChatBot

app = Flask(__name__)

jerry_chat_bot = JerryChatBot()

jerry_chat_bot_convos = []

@app.route('/', methods=['POST', 'GET'])
def home():
    
    if request.method == 'POST':
        print(request.method)
        print(request.form)

        if request.form.get('Clear') == 'Clear':

            jerry_chat_bot_convos.clear()

            return render_template('home.html', conversations=jerry_chat_bot_convos)

        if 'sentence' in request.form:
            sent = request.form['sentence']
            jerry_chat_bot_convos.append('You: ' + sent)
            reply = jerry_chat_bot.reply(sent)
            jerry_chat_bot_convos.append('Jerry: ' + reply)

            return render_template('home.html', conversations=jerry_chat_bot_convos)

 
    elif request.method == 'GET':
        return render_template('home.html')


def main():
    jerry_chat_bot.test_run(False)
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    main()



        # print("DELETE")
        #     

        # else:
        #     return render_template('home.html')



        # if 'sentence' not in request.form:
        #     flash('No sentence post')
        #     redirect(request.url)

        # elif request.form['sentence'] == '':
        #     flash('No sentence')
        #     redirect(request.url)
        # if request.form['submit_button'] == 'Do Something':
        #     print("TEST")
        #     jerry_chat_bot_convos.clear()
        # elif request.form['submit_button'] == 'Do Something Else':
        #     print("TEST2")
