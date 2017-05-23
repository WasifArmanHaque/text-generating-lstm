from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, RepeatVector
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random, sys, os, string
from keras.callbacks import ModelCheckpoint

##### COLOR CODES #######
RED   = "\033[1;31m"
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"
#########################


############ READING AND PREPROCESSING TEXT DATA #############
path = 'Hound of the Baskerville.txt'

text = open(path).read().lower()

# treating all punctuation marks as individual words and newlines as @ sign
for p in string.punctuation:
    text = text.replace(p,' '+p+' ')
text = text.replace('\n', ' @ ')
text = text + '*'
text= text.split()

print('corpus length:', len(text))

words = sorted(list(set(text)))
print('# of unique words:', len(words))

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))


# setting maximum length of each timestep
maxlen = 30
step = 5
sentences = []
next_word = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_word.append(text[i + maxlen])

print('nb sequences:', len(sentences))
print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(sentences), len(words)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_word[i]]] = 1

#############################################



################ BUILDING MODEL AND SETTING HYPERPARAMETERS #######################
if len(sys.argv) == 1:
    print('Building model...')
    model = Sequential()
    model.add(LSTM(200, input_shape=(maxlen, len(words))))
    model.add(Dropout(0.2))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    print('Done!')
elif sys.argv[1] == '-r': # add -r flag if you're loading the model with pre-trained weights
    print('loading model ..')
    json_string = open('sherlock_writer_word.json').read()
    model = model_from_json(json_string)
    model.load_weights('sherlock_writer_word.h5')


optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
filepath = 'weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

##################################################################################################


############## TRAINING AND PREDICTION #####################

for iteration in range(1, 100):
    print()
    print('-' * 50)
    print('Iteration', iteration,' / ',100)
    sys.stdout.write(RED)

    # we run model.fit() for 1 epoch only because we're interested in looking at the generated text after each training step
    model.fit(X, y, batch_size=128, nb_epoch=1, callbacks=[checkpoint])

    # dummy pattern to start predicting on
    pattern = 'sherlock ran towards the room . watson was surprised . then he left the place , looking beyond puzzled . the hound leaped at sherlock as watson looked in horror'
    pattern = pattern.split()

    sys.stdout.write(RESET)
    sys.stdout.write(BLUE)
    for word in pattern:
        sys.stdout.write(word)
        sys.stdout.write(' ')
    for i in range(200):
        # print(pattern)
        sample = np.zeros((1, maxlen, len(words)), dtype=np.bool)

        # properly format the pattern list for feeding it into the LSTM
        for t, char in enumerate(pattern):
            sample[0, t, word_indices[char]] = 1

        preds = model.predict(sample, verbose=0)[0]
        word_ind = np.argmax(preds)
        pred_word = indices_word[word_ind]
        temp = pattern[1:]


        temp.append(pred_word)
        pattern = temp

        if pred_word == '@':
            sys.stdout.write('\n')
        else:
            sys.stdout.write(pred_word)

        sys.stdout.write(' ')
        sys.stdout.flush()
    sys.stdout.write(RESET)
    print

####################################################

# saving the model
json_string = model.to_json()
f = open('sherlock_writer_word.json','w')
f.write(json_string)
f.close()
model.save_weights('sherlock_writer_word.h5', overwrite=True)
