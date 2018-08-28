import numpy as np
np.random.seed(1)
import keras
from keras.models import Sequential,load_model
from keras.layers import Embedding,Bidirectional,SimpleRNN,Dense,Dropout
from keras.callbacks import ModelCheckpoint

def parse_training_data():
    # Word mapping, 0 preserved for padding, 1 preserved for <unk>
    word_pop = {}
    word_mapping = {}
    word_indexing = 2
    word_freq_threshold = 1
    # Zero Padding to max_len
    max_len = 39

    # Parse training data
    train_data_path = 'data/training_data.txt'
    train_sep = ' +++$+++ '
    X_train = []
    Y_train = []

    for line in open(train_data_path,'r', encoding='UTF-8'):
        tmp = line.split(train_sep,1)
        Y_train.append(int(tmp[0]))
        X_train.append(tmp[1])

    # Accumalate each word's frequency
    for sentence in X_train:
        for word in sentence[:-1].split(' '):
            if word not in word_pop:
                word_pop[word] = 1
            else:
                word_pop[word] += 1

    # Map words with frequency > threshold to index, otherwise to 1
    for k in sorted(word_pop.keys()):
        if word_pop[k] > word_freq_threshold:
            word_mapping[k] = word_indexing
            word_indexing += 1
        else:
            word_mapping[k] = 1
    print('Total',word_indexing,'words mapped into index')

    # Save word mapping
    with open('params/word_mapping.txt', 'w') as txt_file:
        for key in word_mapping:
            txt_file.write('%s %s\n' % (key, word_mapping[key]))

    # Transform sentences into sequence of index
    mapped_X_train = []
    for sentence in X_train:
        tmp = []
        for word in sentence[:-1].split(' '):
            tmp.append(word_mapping[word])
        if len(tmp)<max_len:
            tmp.extend([0]*(max_len-len(tmp)))
        elif len(tmp) > max_len:
            tmp = tmp[:max_len]
        mapped_X_train.append(tmp)
    X_train = mapped_X_train

    return X_train, Y_train, word_mapping, word_indexing

def parse_testing_data(word_mapping):
    # Parse testing data
    test_data_path = 'data/testing_data.csv'
    test_sep = ','
    Y_test = []
    X_test = []
    max_len = 39
    for line in open(test_data_path,'r', encoding='UTF-8'):
        tmp = line.split(test_sep,1)
        if tmp[0] == 'label':
            continue
        Y_test.append(int(tmp[0]))
        sentence = tmp[1]
        tmp = []
        for word in sentence[:-1].split(' '):
            if word not in word_mapping:
                tmp.append(1)
            else:
                tmp.append(word_mapping[word])
        if len(tmp)<max_len:
            tmp.extend([0]*(max_len-len(tmp)))
        elif len(tmp) > max_len:
            tmp = tmp[:max_len]
        X_test.append(tmp)

    return X_test, Y_test

def train(X_train, Y_train, word_indexing):
    # Model
    model = Sequential()
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # Set check point to early stop and save the best model
    check = ModelCheckpoint('params/rnn.model', monitor='val_acc', verbose=0, save_best_only=True)

    # TODO, 1-a
    # Embed the words
    model.add(Embedding(word_indexing,256,trainable=True))

    # TODO, 1-b~d
    # Define model architecture
    model.add(Bidirectional(SimpleRNN(128,return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(SimpleRNN(128,return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))

    # TODO, 2
    # Compile model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # TODO, 3
    # Start training
    model.fit(X_train[:50000],Y_train[:50000],batch_size=512,epochs=1,
              callbacks=[check],validation_split=0.1)

def infer(X_test, Y_test):
    # Testing
    model = load_model('params/rnn.model')

    X_test = np.array(X_test)
    loss, acc = model.evaluate(X_test[:10000], Y_test[:10000])

    print('Accuracy : ' + str(acc))

def main():
    X_train, Y_train, word_mapping, word_indexing = parse_training_data()
    X_test, Y_test = parse_testing_data(word_mapping)
    train(X_train, Y_train, word_indexing)
    infer(X_test, Y_test)

if __name__ == '__main__':
    main()
