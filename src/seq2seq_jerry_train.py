import pandas as pd 
import numpy as np 
import argparse

import re

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint

class Seq2Seq_Train_Jerry(object):
    '''
    Character-based seq2seq model that takes in a text file of Question - Answer pairs,
    and trains a seq2seq (endcoder - decoder) model on said pairs.
    Model weights are saved as output files, as are needed one-hot index to char dictionaries 
    and stats about the model
    '''

    def __init__(self):
        # paths to input txt file and to output weights
        # self.txt_file_path = 'data/jerry_q_a.txt'
        # self.weights_file_path = 'models/jerry/jerry_char-weights.h5'
        
        # variables for training the LSTM seq2seq model
        self.batch_size = 64
        self.num_epochs = 100
        self.num_hidden_nodes = 256

        # variables for formatting the Question and Answer text
        self.num_samples = 1000
        self.max_seq_length = 20
    
    def _clean_text(self, input_text, target_text):
        punct_to_remove = '''[#$%&\()*+-/:;<=>@[\\]^_`{|}~]'''
        
        input_text = re.sub("[\(\[].*?[\)\]]", "", input_text).lstrip()
        input_text = ''.join(ch for ch in input_text if ch not in punct_to_remove)
        sent = ""
        for char in input_text:
            if char != '!' and char != '?' and char != '.':
                sent += char
            else:
                sent += char
                break

        input_text = sent[:self.max_seq_length]

        target_text = re.sub("[\(\[].*?[\)\]]", "", target_text).lstrip()
        target_text = ''.join(ch for ch in target_text if ch not in punct_to_remove)
        sent = ""
        for char in target_text:
            if char != '!' and char != '?' and char != '.':
                sent += char
            else:
                sent += char
                break

        target_text = sent[:self.max_seq_length]

        return input_text, target_text

    def load_parse_txt(self, txt_file_path):
        self.txt_file_path = txt_file_path

        self.input_texts = []
        self.target_texts = []
        self.input_characters = set()
        self.target_characters = set()

        # Open txt file, read in lines, and vectorize the data.
        with open(self.txt_file_path, 'r', encoding='utf8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(self.num_samples, len(lines) - 1)]:
            input_text, target_text = line.split('\t')
            
            # clean the text of excess punctuation
            input_text, target_text = self._clean_text(input_text, target_text)

            # using "tab" as the "start sequence" character for the targets
            # using "\n" as "end sequence" character for the targets
            target_text = '\t' + target_text + '\n'
            self.input_texts.append(input_text)
            self.input_texts.append(input_text)
            self.input_texts.append(input_text)
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            self.target_texts.append(target_text)
            self.target_texts.append(target_text)
            self.target_texts.append(target_text)

            # add unique chars to input and output sets
            for char in input_text:
                if char not in self.input_characters:
                    self.input_characters.add(char)
            for char in target_text:
                if char not in self.target_characters:
                    self.target_characters.add(char)
                
        self.input_characters = sorted(list(self.input_characters))
        self.target_characters = sorted(list(self.target_characters))

        # calculate stats for this dataset of inputs and targets
        self.num_inputs = len(self.input_texts)
        self.num_targets = len(self.target_texts)
        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])   
        self.avg_encoder_seq_length = np.mean([len(txt) for txt in self.input_texts])
        self.avg_decoder_seq_length = np.mean([len(txt) for txt in self.target_texts])

        # printing stats for now
        print(f"Number of inputs: {self.num_inputs}")
        print(f"Number of targets: {self.num_targets}")
        print(f"Unique encoder tokens: {self.num_encoder_tokens}")
        print(f"Unique decoder tokens: {self.num_decoder_tokens}")
        print(f"Max encoder seq length: {self.max_encoder_seq_length}")
        print(f"Max decoder seq length: {self.max_decoder_seq_length}")
        print(f"Avg encoder seq length: {self.avg_encoder_seq_length}")
        print(f"Avg decoder seq length: {self.avg_decoder_seq_length}")

        # create dict of stats and save to .npy file
        text_stats = {}
        text_stats['num_inputs'] = self.num_inputs
        text_stats['num_targets'] = self.num_targets
        text_stats['num_encoder_tokens'] = self.num_encoder_tokens
        text_stats['num_decoder_tokens'] = self.num_decoder_tokens
        text_stats['max_encoder_seq_length'] = self.max_encoder_seq_length
        text_stats['max_decoder_seq_length'] = self.max_decoder_seq_length
        text_stats['avg_encoder_seq_length'] = self.avg_encoder_seq_length
        text_stats['avg_decoder_seq_length'] = self.avg_decoder_seq_length
   
        np.save('models/jerry/jerry_text_stats.npy', text_stats)

        # create and save dicts of chars to indices and reverse for encoding and decoding one-hot values
        self.input_char2idx = dict([(char, i) for i, char in enumerate(self.input_characters)])
        self.input_idx2char = dict((i, char) for char, i in self.input_char2idx.items())
        np.save('models/jerry/jerry_input_char2idx.npy', self.input_char2idx)
        np.save('models/jerry/jerry_input_idx2char.npy', self.input_idx2char)

        self.target_char2idx = dict([(char, i) for i, char in enumerate(self.target_characters)])
        self.target_idx2char  = dict((i, char) for char, i in self.target_char2idx.items())
        np.save('models/jerry/jerry_target_char2idx.npy', self.target_char2idx)
        np.save('models/jerry/jerry_target_idx2char.npy', self.target_idx2char)

        self._save_actual_q_a_pairs_used()

    def _save_actual_q_a_pairs_used(self):
        outF = open("data/jerry_q_a_USED.txt", "w")
        outF.write(str(self.input_char2idx))
        outF.write("\n")
        outF.write(str(self.target_char2idx))
        outF.write("\n")
        for q, a in zip(self.input_texts, self.target_texts):
            # write line to output file
            outF.write(q)
            outF.write(a)
        outF.close()
        
    def _create_3d_vectors(self):
        self.encoder_input_data = np.zeros(
                                          (self.num_inputs, 
                                           self.max_encoder_seq_length, 
                                           self.num_encoder_tokens),
                                           dtype='float32')
        self.decoder_input_data = np.zeros(
                                          (self.num_inputs, 
                                           self.max_decoder_seq_length, 
                                           self.num_decoder_tokens),
                                           dtype='float32')
        self.decoder_target_data = np.zeros(
                                           (self.num_inputs, 
                                            self.max_decoder_seq_length, 
                                            self.num_decoder_tokens),
                                            dtype='float32')

        # loop to convert each sequence of chars to one-hot encoded 3-d vectors
        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_char2idx[char]] = 1.

            for t, char in enumerate(target_text):
                self.decoder_input_data[i, t, self.target_char2idx[char]] = 1.
                # decoder_target_data is ahead of decoder_input_data by one timestep
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self.decoder_target_data[i, t - 1, self.target_char2idx[char]] = 1.

    def train_model(self, output_file_path):
        self._create_3d_vectors()

        self.weights_file_path = output_file_path
        # self.num_epochs = num_epochs

        # Define encoder model input and LSTM layers and states
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens), name='encoder_inputs')
        encoder = LSTM(self.num_hidden_nodes, return_state=True, name='encoder_LSTM')
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        
        # discard 'encoder_outputs' and only keep the h anc c states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using 'encoder_states' as initial state.
        # We set up our decoder to return full output sequences (Jerry lines)
        # and to return internal states as well. 
        # We don't use the return states in the training model, but we will use them in inference.
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(self.num_hidden_nodes, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        print(self.model.summary())

        checkpoint = ModelCheckpoint(filepath=self.weights_file_path, 
                                     save_best_only=True, 
                                     save_weights_only=True, 
                                     verbose=1)
        
        # Run training and save weights
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) # 'rmsprop' 'adam'
        self.history = self.model.fit([self.encoder_input_data, self.decoder_input_data], 
                        self.decoder_target_data,
                        batch_size=self.batch_size,
                        epochs=self.num_epochs,
                        validation_split=0.2,
                        callbacks=[checkpoint])

        self.model.save_weights('models/jerry/jerry_char-weights_final.h5')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fit a seq2seq model and save the results.')
    parser.add_argument('--txtdata', 
                        default='data/jerry_q_a.txt',  #jerry_q_a_test.txt
                        help='A txt file with input data.')
    parser.add_argument('--savebest', 
                        default='models/jerry/jerry_char-weights_best.h5', 
                        type=str, 
                        help='A file patj to save the best model weights to.')
    parser.add_argument('--savefinal', 
                    default='models/jerry/jerry_char-weights_best.h5', 
                    type=str, 
                    help='A file path to save the final model weights to.')
    args = parser.parse_args()

    seq2seq_Jerry = Seq2Seq_Train_Jerry()
    seq2seq_Jerry.load_parse_txt(args.txtdata)
    seq2seq_Jerry.train_model(args.out)



    
