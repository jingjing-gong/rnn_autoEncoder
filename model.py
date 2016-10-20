from Config import Config
from helper import Vocab
from myRNN import raw_rnn

import helper

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops
import numpy as np
import sys
import os
import argparse
import logging

args=None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training options")
    
    parser.add_argument('--load-config', action='store_true', dest='load_config', default=False)
    parser.add_argument('--weight-path', action='store', dest='weight_path', required=True)
    parser.add_argument('--debug-enable', action='store_true', dest='debug_enable', default=False)
    
    parser.add_argument('--gpu-num', action='store', dest='gpu_num', default=0, type=int)
    parser.add_argument('--train-test', action='store', dest='train_test', default='train', choices=['train', 'test'])
    
    args = parser.parse_args()

class rnn_autoEncoder(object):
  
    def __init__(self, args=args, test=False):
        self.vocab = Vocab()
        self.config=Config()
        
        self.weight_Path = args.weight_path
        if not os.path.exists(self.weight_Path):
            os.makedirs(self.weight_Path)
            
        if args.load_config == False:
            self.config.saveConfig(self.weight_Path+'/config')
            print 'default configuration generated, please specify --load-config and run again.'
            sys.exit()
        else:
            self.config.loadConfig(self.weight_Path+'/config')
        
        self.step_p_epoch = self.load_data(test)
        
        self.add_placeholders()
        self.add_embedding()
        self.fetch_input()
        logits, self.prediction = self.add_model()
        loss = self.add_loss_op(logits)
        self.train_op = self.add_train_op(loss)
        self.loss = loss
        
        MyVars = [v for v in tf.trainable_variables()]
        MyVars_name = [v.name for v in MyVars]
        print MyVars_name
        
    def decoder(self, cell, initial_state, scope=None):
        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:  # time == 0
                next_cell_state = initial_state
                emit_output= tf.ones(tf.shape(initial_state[1])[:1], dtype=tf.int64) * self.vocab.encode(self.vocab.sos) #(batch_size)
            else:
                next_cell_state = cell_state
                
                logits = tf.nn.rnn_cell._linear(cell_output, output_size=len(self.vocab), #(#(batch_size, vocab_size))
                                 bias=True, scope='decode_words_Linear')
                emit_output = tf.arg_max(logits, dimension=1)                           #(batch_size)
            next_input = tf.nn.embedding_lookup(self.embedding, emit_output, name='next_input_lookup') #(batch_size, embed_size)
            elements_finished = tf.equal(emit_output, self.vocab.encode(self.vocab.eos)) #(batch_size)
            elements_finished = tf.logical_or(elements_finished, (time >= self.config.num_steps))
            next_loop_state = loop_state
            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)

        return raw_rnn(cell, loop_fn, scope=scope)
        
    def add_placeholders(self):
        self.ph_encoder_input = tf.placeholder(tf.int32, (None, None), name='ph_encoder_input')
        self.ph_decoder_input = tf.placeholder(tf.int32, (None, None), name='ph_decoder_input')
        self.ph_decoder_label = tf.placeholder(tf.int32, (None, None), name='ph_decoder_label') #(batch_size, num_steps)
        self.ph_input_encoder_len = tf.placeholder(tf.int32, (None,), name='ph_input_encoder_len') #(batch_size)
        self.ph_input_decoder_len = tf.placeholder(tf.int32, (None,), name='ph_input_decoder_len') #(batch_size)
        self.ph_dropout = tf.placeholder(tf.float32, name='ph_dropout')
    
    def add_embedding(self):
        self.embedding = tf.get_variable('Embedding', [len(self.vocab), self.config.embed_size], trainable=True)
    
    def fetch_input(self):
        self.encoder_input = tf.nn.embedding_lookup(self.embedding, self.ph_encoder_input) #(batch_size, num_steps, embed_size)
        self.decoder_input = tf.nn.embedding_lookup(self.embedding, self.ph_decoder_input) #(batch_size, num_steps+1, embed_size)
    
    def add_model(self):
        """
            input_tensor #(batch_size, num_steps, embed_size)
            input_len    #(batch_size)
        """
        encoder_dropout_input = tf.nn.dropout(self.encoder_input, self.ph_dropout, name='encoder_Dropout')
        decoder_dropout_input = tf.nn.dropout(self.decoder_input, self.ph_dropout, name='decoder_Dropout')
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)
        
        _, state = tf.nn.dynamic_rnn(lstm_cell, encoder_dropout_input, self.ph_input_encoder_len, 
                            dtype=tf.float32, swap_memory=True, time_major=False, scope = 'rnn_encode')
        """construct first decoder state"""
        hid_state = state[1]
        """through a sigmoid convert to digital"""
        logic_hid_state = tf.select((tf.sigmoid(hid_state) > 0.5), 
                                    tf.ones_like(hid_state), tf.zeros_like(hid_state))    
        logic_hid_state = hid_state
        
        with tf.variable_scope('decoder') as vscope:
            #state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros_like(hid_state), logic_hid_state)
            state = tf.nn.rnn_cell.LSTMStateTuple(state[0], tf.zeros_like(hid_state))
            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, decoder_dropout_input, self.ph_input_decoder_len,   #(batch_size, time_steps, hidden_size)
                initial_state=state, dtype=tf.float32, swap_memory=True, time_major=False, scope='rnn_decode')
            with tf.variable_scope('rnn_decode'):
                #tf.reshape(self.ph_decoder_label, shape=(-1, 1)) #(batch_size*time_steps, 1)
                outputs = tf.reshape(outputs, shape=(-1, self.config.hidden_size)) #(batch_size*time_steps, hidden_size)
                logits = tf.nn.rnn_cell._linear(outputs, output_size=len(self.vocab), #(#(batch_size*time_steps, hidden_size))
                                         bias=True, scope='decode_words_Linear')
            vscope.reuse_variables()
            outputs_ta, _, _ = self.decoder(lstm_cell, initial_state=state, scope='rnn_decode')
            
            outputs = outputs_ta.pack() #(time_steps, batch_size)
            outputs = tf.transpose(outputs, [1, 0]) #(batch_size, time_steps)
            
#         self.logic_hid_state = logic_hid_state
        
        return logits, outputs
    
    def add_loss_op(self, logits):
        def seq_loss(logits_tensor, label_tensor, length_tensor):
            """
            Args
                logits_tensor: shape (batch_size*time_steps, hidden_size)
                label_tensor: shape (batch_size, time_steps), label id 1D tensor
                length_tensor: shape(batch_size)
            Return
                loss: A scalar tensor, mean error
            """
    
            labels = tf.reshape(label_tensor, shape=(-1,))
            loss_flat = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='sparse_softmax')
            losses = tf.reshape(loss_flat, shape=tf.shape(label_tensor)) #(batch_size, time_steps)
            length_mask = tf.sequence_mask(length_tensor, tf.shape(losses)[1], dtype=tf.float32, name='length_mask')
            losses_sum = tf.reduce_sum(losses*length_mask, reduction_indices=[1]) #(batch_size)
            losses_mean = losses_sum / tf.to_float(length_tensor) #(batch_size)
            loss = tf.reduce_mean(losses_mean) #scalar
            
            reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v != self.embedding])
            return loss + reg_loss*self.config.reg
        
        return seq_loss(logits, self.ph_decoder_label, self.ph_input_decoder_len)
    
    def add_train_op(self, loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.config.lr, global_step,
                                                   int(self.config.decay_epoch * self.step_p_epoch), self.config.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    
    """data related"""
    def load_data(self, test):
        self.vocab.load_vocab_from_file(self.config.vocab_path)
        if not test:
            self.train_data = helper.mkDataSet(self.config.train_data, self.vocab)
            self.val_data = helper.mkDataSet(self.config.val_data, self.vocab)
            step_p_epoch = len(self.train_data[0]) // self.config.batch_size
        else:
            self.test_data = helper.mkDataSet(self.config.test_data, self.vocab)
            step_p_epoch = 0
        return step_p_epoch

    def create_feed_dict(self, input_batch, encoder_len=None, decoder_len=None, mode='train'):
        """
        note that the order of value in input_batch tuple matters 
        Args
            input_batch, tuple (encoder_input, decoder_input, decoder_label)
            encoder_len, a length list shape of (batch_size)
            decoder_len, a length list shape of (batch_size+1) with one more word <sos> or <eos>
        Returns
            feed_dict: a dictionary that have elements
        """
        if mode == 'train':
            placeholders = (self.ph_encoder_input, self.ph_decoder_input, self.ph_decoder_label, 
                            self.ph_input_encoder_len, self.ph_input_decoder_len, self.ph_dropout)
            data_batch = input_batch + (encoder_len, decoder_len, self.config.dropout)
        elif mode == 'predict':
            placeholders = (self.ph_encoder_input, self.ph_input_encoder_len, self.ph_dropout)
            data_batch = (input_batch[0], encoder_len, self.config.dropout)
        
        feed_dict = dict(zip(placeholders, data_batch))
        
        return feed_dict

    def run_epoch(self, sess, input_data, verbose=None):
        """
        Runs an epoch of training.

        Trains the model for one-epoch.

        Args:
            sess: tf.Session() object
            input_data: tuple of (encode_input, decode_input, decode_label)
        Returns:
            avg_loss: scalar. Average minibatch loss of model on epoch.
        """
        data_len = len(input_data[0])
        total_steps =data_len // self.config.batch_size
        total_loss = []
        for step, (data_batch, lengths_batch) in enumerate(helper.data_iter(*(input_data + (self.config.batch_size, self.vocab)))):
            feed_dict = self.create_feed_dict(data_batch, lengths_batch[0], lengths_batch[1])
            _, loss, lr = sess.run([self.train_op, self.loss, self.learning_rate], feed_dict=feed_dict)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}, lr = {}'.format(
                    step, total_steps, np.mean(total_loss[-verbose:]), lr))
                sys.stdout.flush()
        sys.stdout.write('\n')
        avg_loss = np.mean(total_loss)
        return avg_loss
    
    def fit(self, sess, input_data, verbose=None):
        """
        Runs an epoch of validation or test. return test error

        Args:
            sess: tf.Session() object
            input_data: tuple of (encode_input, decode_input, decode_label)
        Returns:
            avg_loss: scalar. Average minibatch loss of model on epoch.
        """
        data_len = len(input_data[0])
        total_steps =data_len // self.config.batch_size
        total_loss = []
        for step, (data_batch, lengths_batch) in enumerate(helper.pred_data_iter(*(input_data + (self.config.batch_size, self.vocab)))):
            feed_dict = self.create_feed_dict(data_batch, lengths_batch[0], lengths_batch[1])
            loss = sess.run(self.loss, feed_dict=feed_dict)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss[-verbose:])))
                sys.stdout.flush()
        avg_loss = np.mean(total_loss)
        return avg_loss
    
    def predict(self, sess, input_data, verbose=None):
        preds = []
        for _, (data_batch, lengths_batch) in enumerate(helper.pred_data_iter(*(input_data + (self.config.batch_size, self.vocab)))):
            feed_dict = self.create_feed_dict(data_batch, lengths_batch[0], lengths_batch[1], mode='predict')
            pred = sess.run(self.prediction, feed_dict=feed_dict)
            preds+=pred.tolist()
        return preds

def test_case(sess, model, data, onset='VALIDATION'):
    """pred must be list"""
    print '#'*20, 'ON '+onset+' SET START ', '#'*20
    loss = model.fit(sess, data)
    pred = model.predict(sess, data)
    
    decode_max_len = np.max([len(i) for i in data[2]])
    _, decode_label_batch = helper.encodeNpad(data[2], model.vocab, trunLen=decode_max_len)
    accuracy = helper.calculate_accuracy_seq(pred, decode_label_batch, eos_id=model.vocab.encode(model.vocab.eos))
    
    print 'Overall '+onset+' loss is: {}'.format(loss)
    print 'Overall '+onset+' accuracy is: {}'.format(accuracy)
    logging.info('Overall '+onset+' loss is: {}'.format(loss))
    logging.info('Overall '+onset+' accuracy is: {}'.format(accuracy))
    print '#'*20, 'ON '+onset+' SET END ', '#'*20
        
    return loss, pred
  
def train_run():
    logging.info('Training start')
    with tf.Graph().as_default():
        with tf.device("/gpu:" + str(args.gpu_num)):
            model = rnn_autoEncoder()
        saver = tf.train.Saver()
        
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:

            best_loss = np.Inf
            best_val_epoch = 0
            sess.run(tf.initialize_all_variables())
            
            for epoch in range(model.config.max_epochs):
                print "="*20+"Epoch ", epoch, "="*20
                loss = model.run_epoch(sess, model.train_data, verbose=10)
                
                print "Mean loss in this epoch is: ", loss
                logging.info("Mean loss in {}th epoch is: {}".format(epoch, loss))
                
                val_loss, _ = test_case(sess, model, model.val_data, onset='VALIDATION')
                
                if best_loss > val_loss:
                    best_loss = val_loss
                    best_val_epoch = epoch
                    if not os.path.exists(model.weight_Path):
                        os.makedirs(model.weight_Path)

                    saver.save(sess, model.weight_Path+'/parameter.weight')
                if epoch - best_val_epoch > model.config.early_stopping:
                    logging.info("Normal Early stop")
                    break
    logging.info("Training complete")
    
def test_run():
    with tf.Graph().as_default():
        with tf.device("/gpu:" + str(args.gpu_num)):
            model = rnn_autoEncoder(test='test')
        saver = tf.train.Saver()
        
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, model.weight_Path+'/parameter.weight')
            _, pred = test_case(sess, model, model.test_data, onset='TEST')
            
            decode_max_len = np.max([len(i) for i in model.test_data[2]])
            _, decode_label_batch = helper.encodeNpad(model.test_data[2], model.vocab, trunLen=decode_max_len)
            helper.print_pred_seq(pred, decode_label_batch, vocab=model.vocab)
            
def main(_):
    logging.basicConfig(filename=args.weight_path+'/run.log', format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
    if args.train_test == "train":
        train_run()
    else:
        test_run()

if __name__ == '__main__':
    tf.app.run()


    
