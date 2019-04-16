import numpy as np 
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint


class JerryChatBot(object):
    '''
    Character-based seq2seq model that uses pretrained weights to allow the user to have a conversation
    with the Jerry bot
    '''

    def __init__(self):
        # use same num hidden nodes as training model
        self.num_hidden_nodes = 256

        # load in model feature values 
        self.text_stats = np.load('models/jerry/jerry_text_stats.npy').item()
        self.num_encoder_tokens = self.text_stats['num_encoder_tokens']
        self.num_decoder_tokens = self.text_stats['num_decoder_tokens']
        self.max_encoder_seq_length = self.text_stats['max_encoder_seq_length']
        self.max_decoder_seq_length = self.text_stats['max_decoder_seq_length']

        # load in char to idx dictionary values
        self.input_char2idx = np.load('models/jerry/jerry_input_char2idx.npy').item()
        self.input_idx2char = np.load('models/jerry/jerry_input_idx2char.npy').item()
        self.target_char2idx = np.load('models/jerry/jerry_target_char2idx.npy').item()
        self.target_idx2char = np.load('models/jerry/jerry_target_idx2char.npy').item()

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

        self.model.load_weights('models/jerry/jerry_char-weights_test.h5')

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
    
    def _sample_with_diversity(self, preds, temperature=0.5):
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

    def test_run(self):
        input_sentence_1 = "Who are you?"
        input_sentence_2 = "Holy cow!"
        input_sentence_3 = "What's up?"
        input_sentence_4 = "Ask her about Kramer."
        print(f"Input Sentence #1: {input_sentence_1}")
        print(f"Reply #1: {self.reply(input_sentence_1, diversity=False)}")
        print(f"Input Sentence #2: {input_sentence_2}")
        print(f"Reply #2: {self.reply(input_sentence_2)}")
        print(f"Input Sentence #3: {input_sentence_3}")
        print(f"Reply #3: {self.reply(input_sentence_3)}")
        print(f"Input Sentence #4: {input_sentence_4}")
        print(f"Reply #4: {self.reply(input_sentence_4)}")



def main():
    model = JerryChatBot()
    model.test_run()

if __name__ == "__main__":
    main()

