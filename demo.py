import numpy as np
np.random.seed(1)
import keras
from keras.models import Sequential,load_model
from keras.layers import Embedding,Bidirectional,SimpleRNN,Dense,Dropout
from keras.callbacks import ModelCheckpoint

def load_word_mapping(word_mapping_path):
    # Load word mapping
    word_mapping = {}
    for line in open(word_mapping_path, 'r', encoding='UTF-8'):
        word, index = line.split()
        word_mapping[word] = int(index)
    return word_mapping

def parse_input_sentence(input_sentence, word_mapping):
    # Parse input sentence
    max_len = 39

    word_sequence = []
    for word in input_sentence.split(' '):
        if word not in word_mapping:
            word_sequence.append(1)
        else:
            word_sequence.append(word_mapping[word])
    if len(word_sequence)<max_len:
        word_sequence.extend([0]*(max_len-len(word_sequence)))
    elif len(word_sequence) > max_len:
        word_sequence = word_sequence[:max_len]

    return np.expand_dims(word_sequence, axis=0)

def main():
    
    model = load_model('params/rnn.model')
    word_mapping = load_word_mapping('params/word_mapping.txt')
    
    print('\nPlease input some words.')
    input_sentence = input()

    while True:
        word_sequence = parse_input_sentence(input_sentence, word_mapping)
        label_predict = model.predict(word_sequence,batch_size=1,verbose=0)
        print('The predicted value is', label_predict[0])

        if label_predict[0] >= 0.5:
            print('The sentiment of input sentence is positive :)\n')
        else:
            print('The sentiment of input sentence is negative :(\n')
            
        print('Please input some words.')
        input_sentence = input()

if __name__ == '__main__':
    main()
