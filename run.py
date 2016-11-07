import logging

import tensorflow as tf
from keras import backend as K

log = logging.getLogger('')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

from resnets import resnet


def main():
    dataset = 'imagenet'
    # dataset = 'cifar10'
    
    import time
    start = time.time()
    model = resnet.build_model(dataset)
    duration = time.time() - start
    print('{} s to make model'.format(duration))

    start = time.time()
    model.output
    duration = time.time() - start
    print('{} s to get output'.format(duration))

    model.summary()

    # start = time.time()
    # model.compile(loss='categorical_crossentropy', optimizer='sgd')
    # duration = time.time() - start
    # print('{} s to compile'.format(duration))

if __name__ == '__main__':
    main()
    
