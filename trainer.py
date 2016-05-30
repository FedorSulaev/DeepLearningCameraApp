import functools
import glob
import itertools
import multiprocessing
import random
import sys
import time
import cv2
import numpy
import tensorflow as tf

import data_generator
import model
import constants

def number_to_vec(p, number):
    def char_to_vec(c):
        y = numpy.zeros((len(constants.CHARS),))
        y[constants.CHARS.index(c)] = 1.0
        return y
    c = numpy.vstack([char_to_vec(c) for c in number])
    return numpy.concatenate([[1. if p else 0], c.flatten()])
    
def read_data(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        number = fname.split("/")[1][9:16]
        p = fname.split("/")[1][17] == '1'
        yield im, number_to_vec(p, number)
        
def unzip(b):
    xs, ys = zip(*b)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys
    
def batch(it, batch_size):
    out = []
    for x in it:
        out.append(x)
        if len(out) == batch_size:
            yield out
            out = []
    if out:
        yield out
        
def mpgen(f):
    def main(q, args, kwargs):
        try:
            for item in f(*args, **kwargs):
                q.put(item)
        finally:
            q.close()
            
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        q = multiprocessing.queues.Queue(3)
        proc = multiprocessing.process(target=main, args=(q, args, kwargs))
        proc.start()
        try:
            while True:
                item = q.get()
                yield item
        finally:
            proc.terminate()
            proc.join()
    return wrapped
    
@mpgen
def read_batches(batch_size):
    def gen_vecs():
        for im, c, p in data_generator.generate_ims(batch_size):
            yield im, number_to_vec(p, c)
            
    while True:
        yield unzip(gen_vecs())
        
def train(learn_rate, report_steps, batch_size, initial_weights=None):
    x, y, params = model.get_training_model()
    y_ = tf.placeholder(tf.float32, [None, 7 * len(constants.CHARS) + 1])
    digits_loss = tf.nn.softmax_cross_entropy_with_logits(
        tf.reshape(y[:, 1:], [-1, len(constants.CHARS)]),
        tf.reshape(y_[:, 1:], [-1, len(constants.CHARS)]))
    digits_loss = tf.reduce_sum(digits_loss)
    presence_loss = 10. * tf.nn.sigmoid_cross_entropy_with_logits(
        y[:, :1], y_[:, :1])
    presence_loss = tf.reduce_sum(presence_loss)
    cross_entropy = digits_loss + presence_loss
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)
    
    best = tf.argmax(tf.reshape(y[:, 1:], [-1, 7, len(constants.CHARS)]), 2)
    correct = tf.argmax(tf.reshape(y_[:, 1:], [-1, 7, len(constants.CHARS)]), 2)
    
    if initial_weights is not None:
        assert len(params) == len(initial_weights)
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]
        
    init = tf.initialize_all_variables()
    
    def vec_to_number(v):
        return "".join(constants.CHARS[i] for i in v)
        
    def do_report():
        r = sess.run([best, correct, tf.greater(y[:, 0], 0),
                      y_[:, 0], digits_loss, presence_loss, cross_entropy],
                        feed_dict={x: test_xs, y_: test_ys})
        num_correct = numpy.sum(numpy.logical_or(
            numpy.all(r[0] == r[1], axis=1),
            numpy.logical_and(r[2] < 0.5, r[3] < 0.5)))
        r_short = (r[0][:190], r[1][:190], r[2][:190], r[3][:190])
        for b, c, pb, pc in zip(*r_short):
            print("{} {} <-> {} {}".format(vec_to_number(c), pc,
                                            vec_to_number(b), float(pb)))
        num_p_correct = numpy.sum(r[2] == r[3])
        
        print("B{:3d} {:2.02f}% {:02.02f}% loss: {} "
                "(digits: {}, presence: {}) |{}|".format(
                batch_idx, 100. * num_correct / (len(r[0])),
                100. * num_p_correct / len(r[2]), r[6], r[4], r[5],
                "".join("X "[numpy.array_equal(b, c) or (not pb and not pc)]
                    for b, c, pb, pc in zip(*r_short))))
                        
    def do_batch():
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if batch_idx % report_steps == 0:
            do_report()
            
    with tf.Session as sess:
        sess.run(init)
        if initial_weights is not None:
            sess.run(assign_ops)

        test_xs, test_ys = unzip(list(read_data("test/*.png"))[:50])

        try:
            last_batch_idx = 0
            last_batch_time = time.time()
            batch_iter = enumerate(read_batches(batch_size))
            for batch_idx, (batch_xs, batch_ys) in batch_iter:
                do_batch()
                if batch_idx % report_steps == 0:
                    batch_time = time.time()
                    if last_batch_idx != batch_idx:
                        print("time for 60 batches {}".format(
                            60 * (last_batch_time - batch_time) /
                                            (last_batch_idx - batch_idx)))
                        last_batch_idx = batch_idx
                        last_batch_time = batch_time

        except KeyboardInterrupt:
            last_weights = [p.eval() for p in params]
            numpy.savez("weights.npz", *last_weights)
            return last_weights