# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 18:18:34 2016

@author: Fedor Sulaev
"""

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))