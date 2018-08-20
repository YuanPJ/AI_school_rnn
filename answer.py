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
    tr_path = 'data/training_label.txt'
    tr_sep = ' +++$+++ '
    tr_label = []
    tr_data = []

    for line in open(tr_path,'r'):
        tmp = line.split(tr_sep,1)
        tr_label.append(int(tmp[0]))
        tr_data.append(tmp[1])

    # Accumalate each word's frequency
    for sentence in tr_data:
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

    # Transform sentences into sequence of index
    mapped_tr_data = []
    for sentence in tr_data:
        tmp = []
        for word in sentence[:-1].split(' '):
            tmp.append(word_mapping[word])
        if len(tmp)<max_len:
            tmp.extend([0]*(max_len-len(tmp)))
        elif len(tmp) > max_len:
            tmp = tmp[:max_len]
        mapped_tr_data.append(tmp)
    tr_data = mapped_tr_data

    return tr_data, tr_label, word_mapping, word_indexing

def parse_testing_data(word_mapping):
    # Parse testing data
    tt_path = 'data/testing_data.csv'
    tt_sep = ','
    tt_solution = []
    tt_data = []
    max_len = 39
    for line in open(tt_path,'r'):
        tmp = line.split(tt_sep,1)
        if tmp[0] == 'label':
            continue
        tt_solution.append(int(tmp[0]))
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
        tt_data.append(tmp)

    return tt_data, tt_solution

def train(tr_data, tr_label, word_indexing):
    # Model
    model = Sequential()
    tr_data = np.array(tr_data)
    tr_label = np.array(tr_label)

    # Set check point to early stop and save the best model
    check = ModelCheckpoint('SampleSolution.model', monitor='val_acc', verbose=0, save_best_only=True)

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
    model.fit(tr_data,tr_label,batch_size=512,epochs=5,
              callbacks=[check],validation_split=0.1)

def test(tt_data, tt_solution):
    # Testing
    model = load_model('SampleSolution.model')

    tt_data = np.array(tt_data)
    tt_pred = model.predict(tt_data,batch_size=512,verbose=1)

    # Calculating accuracy
    score = 0
    for idx in range(len(tt_pred)):
        label = 1 if tt_pred[idx] > .5 else 0
        if label == tt_solution[idx]:
            score += 1
    accuracy = score / len(tt_pred)
    print('Testing accuracy:', accuracy)

def main():
    tr_data, tr_label, word_mapping, word_indexing = parse_training_data()
    tt_data, tt_solution = parse_testing_data(word_mapping)
    train(tr_data, tr_label, word_indexing)
    test(tt_data, tt_solution)

if __name__ == '__main__':
    main()
