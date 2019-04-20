import numpy as np 
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint
import argparse


class JerryChatBot(object):
    '''
    Character-based seq2seq model that uses pretrained weights to allow the user to have a conversation.
    with the Jerry bot.
    Uses argument parsing to specify the model to use for predictions, and for continuous chat mode.
    If you would like to chat continuously when running, set chat=True in argument parsing.
    '''

    def __init__(self, model_to_use, best_or_final):  #'models/jerry/samples_5000_seq_40/' #'final'
        # use same num hidden nodes as training model
        self.num_hidden_nodes = 256

        # load in model feature values 
        self.text_stats = np.load(model_to_use + 'jerry_text_stats.npy').item()
        self.num_encoder_tokens = self.text_stats['num_encoder_tokens']
        self.num_decoder_tokens = self.text_stats['num_decoder_tokens']
        self.max_encoder_seq_length = self.text_stats['max_encoder_seq_length']
        self.max_decoder_seq_length = self.text_stats['max_decoder_seq_length']

        # load in char to idx dictionary values
        self.input_char2idx = np.load(model_to_use + 'jerry_input_char2idx.npy').item()
        self.input_idx2char = np.load(model_to_use + 'jerry_input_idx2char.npy').item()
        self.target_char2idx = np.load(model_to_use + 'jerry_target_char2idx.npy').item()
        self.target_idx2char = np.load(model_to_use + 'jerry_target_idx2char.npy').item()

        # Define encoder model input and LSTM layers and states exactly as defined in training model
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens), name='encoder_inputs')
        encoder = LSTM(self.num_hidden_nodes, return_state=True, name='encoder_LSTM')
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(self.num_hidden_nodes, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # load in either the best weights or final weights from the trained model.
        # in general, the final weights give more varied answers/responses, as the
        # "best" model seems to answer the same no matter the input. Still unsure why
        # this 
        if best_or_final == 'final':
            file_path = model_to_use + 'jerry_char-weights_final_so_' + model_to_use[-14:-8] + model_to_use[-4:-1] + '.h5'
            self.model.load_weights(file_path)
        
        else:
            file_path = model_to_use + 'jerry_char-weights_best_so_' + model_to_use[-14:-8] + model_to_use[-4:-1] + '.h5'
            self.model.load_weights(file_path)

        # create encoder and decoder models for prediction
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.num_hidden_nodes,))
        decoder_state_input_c = Input(shape=(self.num_hidden_nodes,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def _encode_input_sentence(self, sentence):
        if len(sentence) > self.max_encoder_seq_length:
            sentence = sentence[:self.max_encoder_seq_length]

        encoder_input_sent = np.zeros((1, 
                                       self.max_encoder_seq_length, 
                                       self.num_encoder_tokens), 
                                       dtype='float32')
    
        for t, char in enumerate(sentence):
            encoder_input_sent[0, t, self.input_char2idx[char]] = 1
        
        return encoder_input_sent
    
    def _sample_with_diversity(self, preds, temperature=0.314532):  #0.453212 #0.2212
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        
        return np.argmax(probas)

    def reply(self, sentence, diversity=False):
        self.encoder_input_sent = self._encode_input_sentence(sentence)

        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(self.encoder_input_sent)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_char2idx['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            if diversity:
                sampled_token_index = self._sample_with_diversity(output_tokens[0, -1, :])

            else:
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
            
            sampled_char = self.target_idx2char[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def test_run(self, chat, diversity):
        input_sentence_1 = "Do you know?"
        input_sentence_2 = "Do you know?"
        input_sentence_3 = "Ha."
        input_sentence_4 = "Wait wait wait, what"
        print(f"Input Sentence #1: {input_sentence_1}")
        print(f"Reply #1: {self.reply(input_sentence_1, diversity=True)}")
        print(f"Input Sentence #2: {input_sentence_2}")
        print(f"Reply #2: {self.reply(input_sentence_2)}")
        print(f"Input Sentence #3: {input_sentence_3}")
        print(f"Reply #3: {self.reply(input_sentence_3)}")
        print(f"Input Sentence #4: {input_sentence_4}")
        print(f"Reply #4: {self.reply(input_sentence_4)}")

        if chat:
            self._chat_over_command_line(diversity)

    def _chat_over_command_line(self, diversity):
        print("Welcome to the Jerry ChatBot!")
        print("Please type anything, and Jerry will respond!")
        print("Type 'exit' to stop")

        user_sent = input("User: " )

        while user_sent != 'exit':
            try:
                print(f"Jerry: {self.reply(user_sent, diversity=diversity)}") #[:-1]
            except:
                print(f"What's the deal with that sentence?! \nPlease try something else!")
            user_sent = input("User: " )

        exit()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(
    description='Predict Jerry Seinfeld dialogue using seq2seq model.')
    parser.add_argument('--model', 
                        default='models/jerry/samples_16000_seq_100/',  #jerry_q_a_test.txt
                        help='location of model files to use to predict')
    parser.add_argument('--bestorfinal', 
                        default='final',  #jerry_q_a_test.txt
                        help='use best model weights or final model weights for prediction')
    parser.add_argument('--chat', 
                        default=True, 
                        type=str2bool, 
                        nargs='?',
                        const=True, #jerry_q_a_test.txt
                        help='continuously chat with JerryBot?')
    parser.add_argument('--chatwithdiversity', 
                        default=False, 
                        type=str2bool, 
                        nargs='?',
                        const=False, #jerry_q_a_test.txt
                        help='chat with diversity with JerryBot?')

    args = parser.parse_args()

    model = JerryChatBot(args.model, args.bestorfinal)
    model.test_run(args.chat, args.chatwithdiversity)

if __name__ == "__main__":
    main()

