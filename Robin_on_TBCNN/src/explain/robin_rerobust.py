import sys
sys.path.append("../")
import os
os.environ[ 'TF_KERAS']= '1'
import argparse
import random
import pickle
import tensorflow.compat.v1 as tf
import data_loader.base_data_loader #BaseDataLoader
import util.data.data_loader.base_data_loader #BaseDataLoader
from util.threaded_iterator import ThreadedIterator
import re
import copy
import time
import argument_parser
import copy
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from datetime import datetime
from keras_radam.training import RAdamOptimizer
import logging
from explainer_robin_rerobust import TBCNN_explainer
from util.network.tbcnn import TBCNN
from util import util_functions
logging.basicConfig(filename='training.log',level=logging.DEBUG)
np.set_printoptions(threshold=sys.maxsize)
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()


def predict_class(batch_data, train_opt):
    #g1_train_opt = train_opt
    #g1_train_opt.conv_output_dim = 100
    g1 = tf.Graph()
    modelpath = "../model/tbcnn_parser_pycparser-type_100-token_100-conv_output_100-node_init_2-num_conv_4"
    # ../model/tbcnn_parser_pycparser-type_100-token_100-conv_output_100-node_init_2-num_conv_4
    checkfile1 = os.path.join(modelpath, 'cnn_tree.ckpt')
    ckpt1 = tf.train.get_checkpoint_state(modelpath)
    #print("The model path : " + str(checkfile1))
    #if ckpt1 and ckpt1.model_checkpoint_path:
        #print("-------Continue training with old model-------- : " + str(checkfile1))

    sess1 = tf.Session(graph=g1)
    with sess1.as_default(): 
        with g1.as_default():
            original_model = TBCNN(train_opt)
            original_model.feed_forward()

            saver1 = tf.train.Saver(save_relative_paths=True, max_to_keep=5)  
            init = tf.global_variables_initializer()
    predict_onehot1 = []
    with sess1.as_default():
        with sess1.graph.as_default():
            sess1.run(init)
            if ckpt1 and ckpt1.model_checkpoint_path:
                saver1.restore(sess1, ckpt1.model_checkpoint_path)
            
            correct_labels1 = []
            predictions1 = []
            
            scores1 = sess1.run(
                    [original_model.softmax],
                    feed_dict={
                        original_model.placeholders["node_type"]: batch_data["batch_node_type_id"],
                        original_model.placeholders["node_token"]:  batch_data["batch_node_sub_tokens_id"],
                        original_model.placeholders["children_index"]:  batch_data["batch_children_index"],
                        original_model.placeholders["children_node_type"]: batch_data["batch_children_node_type_id"],
                        original_model.placeholders["children_node_token"]: batch_data["batch_children_node_sub_tokens_id"],
                        original_model.placeholders["labels"]: batch_data["batch_labels_one_hot"],
                        original_model.placeholders["dropout_rate"]: 0.0
                    }
                )

            correct_labels1.extend(np.argmax(batch_data["batch_labels_one_hot"],axis=1))
            predictions1.extend(np.argmax(scores1[0],axis=1))
            label_size = 104 #
            predict_onehot1 = []
            for pre in predictions1:
                zeros = np.zeros(label_size)
                zeros[pre] = 1.0
                predict_onehot1.append(zeros)
            #print("correct_labels", correct_labels1)
            #print("predictions", predictions1)
    sess1.close()
        
    return predict_onehot1

def main(train_opt, test_opt):

    train_opt.model_path = os.path.join("../model_rerobust", util_functions.form_tbcnn_model_path(train_opt))
    checkfile = os.path.join(train_opt.model_path, 'cnn_tree.ckpt')
    ckpt = tf.train.get_checkpoint_state(train_opt.model_path)
    print("The model path : " + str(checkfile))
    if ckpt and ckpt.model_checkpoint_path:
        print("-------Continue training with old model-------- : " + str(checkfile))


    tbcnn_model = TBCNN_explainer(train_opt)
    tbcnn_model.feed_forward()
    #print("var_list:", tbcnn_model.var_list)
    #print("pos_var_list:", tbcnn_model.pos_var_list)
    #print("neg_var_list:", tbcnn_model.neg_var_list)
    dif_path = "../dif_file/dif-buckets-train.pkl"
    #dif_path = "../OJ104_pycparser_train_test_val/pycparser-buckets-train.pkl"
    train_data_loader = data_loader.base_data_loader.BaseDataLoader(train_opt.batch_size, train_opt.label_size, train_opt.tree_size_threshold_upper, train_opt.tree_size_threshold_lower, train_opt.train_path, dif_path, True)
    test_data_loader = util.data.data_loader.base_data_loader.BaseDataLoader(test_opt.batch_size, test_opt.label_size, test_opt.tree_size_threshold_upper, test_opt.tree_size_threshold_lower, test_opt.test_path, False)

    optimizer = RAdamOptimizer(train_opt.lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        pos_training_point = optimizer.minimize(tbcnn_model.pos_loss, var_list = tbcnn_model.pos_var_list)
        neg_training_point = optimizer.minimize(tbcnn_model.neg_loss_dif, var_list = tbcnn_model.neg_var_list)
    saver = tf.train.Saver(save_relative_paths=True, max_to_keep=5)  
    init = tf.global_variables_initializer()

    best_f1 = test_opt.best_f1
    with tf.Session() as sess:
        sess.run(init)
        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training with old model")
            print("Checkpoint path : " + str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i, var in enumerate(saver._var_list):
                print('Var {}: {}'.format(i, var))

        
        for epoch in range(1,  train_opt.epochs + 1):
            train_batch_iterator = ThreadedIterator(train_data_loader.make_minibatch_iterator(), max_queue_size=train_opt.worker)
            for train_step, train_batch_data in enumerate(train_batch_iterator):
                print("***************")
                #print(train_batch_data["batch_node_sub_tokens_id"].shape)
                _, err = sess.run(
                        [pos_training_point, tbcnn_model.pos_loss],
                        feed_dict={
                            tbcnn_model.placeholders["node_type"]: train_batch_data["batch_node_type_id"],
                            tbcnn_model.placeholders["node_token"]:  train_batch_data["batch_node_sub_tokens_id"],
                            tbcnn_model.placeholders["children_index"]:  train_batch_data["batch_children_index"],
                            tbcnn_model.placeholders["children_node_type"]: train_batch_data["batch_children_node_type_id"],
                            tbcnn_model.placeholders["children_node_token"]: train_batch_data["batch_children_node_sub_tokens_id"],
                            #tbcnn_model.placeholders["labels"]: train_batch_data["batch_labels_one_hot"],
                            tbcnn_model.placeholders["labels"]: predict_class(train_batch_data, train_opt),
                            tbcnn_model.placeholders["dropout_rate"]: 0.3,
                            tbcnn_model.placeholders["dif_node_type"]: train_batch_data["dif_batch_node_type_id"],
                            tbcnn_model.placeholders["dif_node_token"]:  train_batch_data["dif_batch_node_sub_tokens_id"],
                            tbcnn_model.placeholders["dif_children_index"]:  train_batch_data["dif_batch_children_index"],
                            tbcnn_model.placeholders["dif_children_node_type"]: train_batch_data["dif_batch_children_node_type_id"],
                            tbcnn_model.placeholders["dif_children_node_token"]: train_batch_data["dif_batch_children_node_sub_tokens_id"]
                        }
                    )
                print("Epoch:", epoch, "Step:",train_step,"Loss:", err, "Best F1:", best_f1)
                
                _, err = sess.run(
                        [neg_training_point, tbcnn_model.neg_loss],
                        feed_dict={
                            tbcnn_model.placeholders["node_type"]: train_batch_data["batch_node_type_id"],
                            tbcnn_model.placeholders["node_token"]:  train_batch_data["batch_node_sub_tokens_id"],
                            tbcnn_model.placeholders["children_index"]:  train_batch_data["batch_children_index"],
                            tbcnn_model.placeholders["children_node_type"]: train_batch_data["batch_children_node_type_id"],
                            tbcnn_model.placeholders["children_node_token"]: train_batch_data["batch_children_node_sub_tokens_id"],
                            #tbcnn_model.placeholders["labels"]: train_batch_data["batch_labels_one_hot"],
                            tbcnn_model.placeholders["labels"]: predict_class(train_batch_data, train_opt),
                            tbcnn_model.placeholders["dropout_rate"]: 0.3,
                            tbcnn_model.placeholders["dif_node_type"]: train_batch_data["dif_batch_node_type_id"],
                            tbcnn_model.placeholders["dif_node_token"]:  train_batch_data["dif_batch_node_sub_tokens_id"],
                            tbcnn_model.placeholders["dif_children_index"]:  train_batch_data["dif_batch_children_index"],
                            tbcnn_model.placeholders["dif_children_node_type"]: train_batch_data["dif_batch_children_node_type_id"],
                            tbcnn_model.placeholders["dif_children_node_token"]: train_batch_data["dif_batch_children_node_sub_tokens_id"]
                        }
                    )
                
                
                print("Epoch:", epoch, "Step:",train_step,"Loss:", err, "Best F1:", best_f1)

                if train_step % train_opt.checkpoint_every == 0 and train_step > 0:
                               
                    #Perform Validation
                    print("Perform validation.....")
                    correct_labels = []
                    s_predictions = []
                    us_predictions = []
                    test_batch_iterator = ThreadedIterator(test_data_loader.make_minibatch_iterator(), max_queue_size=test_opt.worker)
                    for test_step, test_batch_data in enumerate(test_batch_iterator):
                        print("***************")

                        #print(test_batch_data["batch_size"])
                        s_scores, us_scores = sess.run(
                                [tbcnn_model.s_softmax, tbcnn_model.us_softmax],
                                feed_dict={
                                    tbcnn_model.placeholders["node_type"]: test_batch_data["batch_node_type_id"],
                                    tbcnn_model.placeholders["node_token"]:  test_batch_data["batch_node_sub_tokens_id"],
                                    tbcnn_model.placeholders["children_index"]:  test_batch_data["batch_children_index"],
                                    tbcnn_model.placeholders["children_node_type"]: test_batch_data["batch_children_node_type_id"],
                                    tbcnn_model.placeholders["children_node_token"]: test_batch_data["batch_children_node_sub_tokens_id"],
                                    #tbcnn_model.placeholders["labels"]: test_batch_data["batch_labels_one_hot"],
                                    tbcnn_model.placeholders["labels"]: predict_class(test_batch_data, train_opt),
                                    tbcnn_model.placeholders["dropout_rate"]: 0.0
                                }
                            )
                        #print(s_scores)
                        batch_correct_labels = list(np.argmax(test_batch_data["batch_labels_one_hot"], axis=1))
                        s_batch_predictions = list(np.argmax(s_scores,axis=1))
                        us_batch_predictions = list(np.argmax(us_scores,axis=1))
                        print(batch_correct_labels)
                        print(s_batch_predictions)
                        print(us_batch_predictions)

                        correct_labels.extend(np.argmax(test_batch_data["batch_labels_one_hot"],axis=1))
                        s_predictions.extend(np.argmax(s_scores,axis=1))
                        us_predictions.extend(np.argmax(us_scores,axis=1))
                    #print(correct_labels)
                    #print(predictions)
                    #s_f1 = float(f1_score(correct_labels, s_predictions, average="micro"))
                    #us_f1 = float(f1_score(correct_labels, us_predictions, average="micro"))
                    s_f1 = float(accuracy_score(correct_labels, s_predictions))
                    us_f1 = float(accuracy_score(correct_labels, us_predictions))
                    print(classification_report(correct_labels, s_predictions))
                    print(classification_report(correct_labels, us_predictions))
                    print('AS-F1:', s_f1)
                    print('AU-F1:', us_f1)
                    print('Best F1:', best_f1)
                    # print(confusion_matrix(correct_labels, predictions))

                    if s_f1 > best_f1:
                        best_f1 = s_f1
                        saver.save(sess, checkfile)                  
                        print('Checkpoint saved, epoch:' + str(epoch) + ', step: ' + str(train_step) + ', loss: ' + str(err) + '.')



if __name__ == "__main__":
    train_opt = argument_parser.parse_arguments()
    
    test_opt = copy.deepcopy(train_opt)
    # test_opt.data_path = "OJ_rs/OJ_rs-buckets-test.pkl"

    os.environ['CUDA_VISIBLE_DEVICES'] = train_opt.cuda

    main(train_opt, test_opt)
   