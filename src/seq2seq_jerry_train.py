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
    Model weights are saved as output files, as are the one-hot index to char dictionaries 
    and stats about the model.
    See below for argument parsing when running from the command line.
    Note the 'history' files can be very large if using a large dataset.
    '''

    def __init__(self):        
        # variables for training the LSTM seq2seq model
        self.batch_size = 64
        self.num_epochs = 500
        self.num_hidden_nodes = 256

        # variables for loading and formatting the Question and Answer text
        self.txt_file_path = None
        self.num_samples = None
        self.max_seq_length = None
        self.duplicate_records = None
        self.sentences_only = None
    
    def _clean_text(self, input_text, target_text):
        '''
        Removes all unneeded punctuation (which would in turn create a larger X matrix).
        Keeps '.!?," punctuation marks, as those adds to semantic meaning. 
        Right now, all uppercase text is included as well, but might change that to lower for testing.
        '''
        punct_to_remove = '''[#$%&\()*+-/:;<=>@[\\]^_`{|}~]'''
        eos_punct = '''!?.'''
        
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
    
    def _clean_sentences(self, input_text_lst, target_text_lst):
        # either keep full sentences or include all text up to max_seq_length,
        # but either way, remove any blank sentences.
        input_text_to_keep = []
        target_text_to_keep = []

        if self.sentences_only:
            eos_punct = '''!?.'''
            eos_punct = ('!', '?', '.')
            for input_text, target_text in zip(input_text_lst, target_text_lst):
                if len(input_text) < 2 or len(target_text) < 2:
                    continue

                if input_text[-1] in eos_punct and target_text[-1] in eos_punct:
                    input_text_to_keep.append(input_text)
                    target_text_to_keep.append(target_text)

        else:
            for input_text, target_text in zip(input_text_lst, target_text_lst):
                if len(input_text) > 1 and len(target_text) > 1:
                    input_text_to_keep.append(input_text)
                    target_text_to_keep.append(target_text)

        return input_text_to_keep, target_text_to_keep

    def load_parse_txt(self, 
                       txt_file_path,
                       num_samples, 
                       max_seq_length, 
                       duplicate_records,
                       sentences_only):
        '''
        Loads in Q-A paris from txt file, and calls functions to clean and procees text, 
        as well as create dictionaries for converting characters to integer values.
        Saves dictionaries and info about the models as .npy output files, so can easily
        load into the predict model when chatting.
        '''

        # sets variables based on argument parsing
        self.txt_file_path = txt_file_path
        self.num_samples = num_samples
        self.max_seq_length = max_seq_length
        self.duplicate_records = duplicate_records
        self.sentences_only = sentences_only

        self.input_texts_initial = []
        self.target_texts_initial = []
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

            self.input_texts_initial.append(input_text)
            self.target_texts_initial.append(target_text)

        # either keep full sentences or include all text up to max_seq length
        # but either way, 
        self.input_texts_initial,  self.target_texts_initial = self._clean_sentences(self.input_texts_initial, self.target_texts_initial)

        for input_text, target_text in zip(self.input_texts_initial, self.target_texts_initial):
            # using "tab" as the "start sequence" character for the targets
            # using "\n" as "end sequence" character for the targets
            target_text = '\t' + target_text + '\n'
            
            #print(self.duplicate_records)
            if self.duplicate_records:
                self.input_texts.append(input_text)
                self.input_texts.append(input_text)
                self.input_texts.append(input_text)
                self.input_texts.append(input_text)
                self.target_texts.append(target_text)
                self.target_texts.append(target_text)
                self.target_texts.append(target_text)
                self.target_texts.append(target_text)
            
            else:
                self.input_texts.append(input_text)
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
        outF = open("models/jerry/jerry_q_a_USED.txt", "w")
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

    def train_model(self, best_weights_file_path, final_weights_file_path):
        self._create_3d_vectors()

        self.best_weights_file_path = best_weights_file_path
        self.final_weights_file_path = final_weights_file_path
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

        checkpoint = ModelCheckpoint(filepath=self.best_weights_file_path, 
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

        self.model.save_weights(self.final_weights_file_path)

        np.save('models/jerry/jerry_model_history.npy', self.history.history)

def str2bool(v):
    # allows for boolean values to be used in argument parsing
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Train a seq2seq model and save the model weight results.')

    # arguments for training model parameters
    parser.add_argument('--txtdata', 
                        default='data/jerry_q_a.txt',  #jerry_q_a_test.txt
                        help='A txt file with input data in question-answer format')
    parser.add_argument('--numsamples', 
                        default=16000, 
                        type=int, 
                        help='The number of input samples to train the model on')
    parser.add_argument('--maxseqlength', 
                        default=20, 
                        type=int, 
                        help='The max sequence length to use when training the model')
    parser.add_argument('--duplicaterecords', 
                        default=True, 
                        type=str2bool, 
                        nargs='?',
                        const=True,
                        help='Choose to oversample input data or not')
    parser.add_argument('--sentencesOnly', 
                        default=False, 
                        type=str2bool, 
                        nargs='?',
                        const=False,
                        help='Choose to only include full sentences or not in training data')
    
    # arguments for model weights file paths
    parser.add_argument('--savebest', 
                        default='models/jerry/jerry_char-weights_best.h5', 
                        type=str, 
                        help='File path to save the best model weights to.')
    parser.add_argument('--savefinal', 
                        default='models/jerry/jerry_char-weights_final.h5', 
                        type=str, 
                        help='File path to save the final model weights to.')
    args = parser.parse_args()

    seq2seq_Jerry = Seq2Seq_Train_Jerry()

    seq2seq_Jerry.load_parse_txt(args.txtdata,
                                 args.numsamples, 
                                 args.maxseqlength, 
                                 args.duplicaterecords,
                                 args.sentencesOnly)
    seq2seq_Jerry.train_model(args.savebest, args.savefinal)