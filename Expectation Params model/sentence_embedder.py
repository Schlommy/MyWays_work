"""
    Author: Puneet Grover
    Version: 0.1.0
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub
import numpy as np
import math

class SentenceEmbedder(object):
  def __init__(self):
    """
    """
    # Create graph and finalize (finalizing optional but recommended).
    self.g = tf.Graph()
    with self.g.as_default():
      # We will be feeding 1D tensors of text into the graph.
      self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
      #self.list_input = tf.placeholder(dtype=tf.string, shape=[1, None])
      self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
      self.embedded_text = self.embed(self.text_input)
      #self.embedded_list = self.embed(self.list_input)
      self.init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    self.g.finalize()
    
    self.session = tf.Session(graph=self.g)
    self.session.run(self.init_op)
  
  def get(self, sentence):
    """
    """
    if type(sentence) != str: sentence = ""
    #if type(sentence) in [list, tuple, np.array]:
    #  return self.session.run(self.embedded_list0, feed_dict={self.list_input: sentence})
    #else:
    return self.session.run(self.embedded_text, feed_dict={self.text_input: [sentence]})[0]
  
  def similarity(self, sentence, lst):
    """
    """
    if type(lst)== str: lst = [lst]
    sent_embedd = self.session.run(self.embedded_text, feed_dict={self.text_input: [sentence]})
    lst_embedd = []
    for s in lst: lst_embedd.append(self.session.run(self.embedded_text, feed_dict={self.text_input: [s]}))
    
    result = []
    from sklearn.metrics.pairwise import cosine_similarity
    
    for emb in lst_embedd:
      result.append(cosine_similarity(sent_embedd, emb)[0][0])
      
    return np.array(result)

