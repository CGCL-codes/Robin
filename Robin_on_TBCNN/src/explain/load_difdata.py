import sys
sys.path.append("../")
import os
os.environ[ 'TF_KERAS']= '1'

from collections import defaultdict
import pickle
import numpy as np

import argument_parser
from data_processor.pycparser_data_processor import PycParserDataProcessor
from util.network.tbcnn import TBCNN
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

#load model
test_opt = argument_parser.parse_arguments()
model_path = '../model/tbcnn_parser_pycparser-type_100-token_100-conv_output_100-node_init_2-num_conv_4'
checkfile = os.path.join(model_path, 'cnn_tree.ckpt')
ckpt = tf.train.get_checkpoint_state(model_path)
tbcnn_model = TBCNN(test_opt)
tbcnn_model.feed_forward()
saver = tf.train.Saver(save_relative_paths=True, max_to_keep=5)  
init = tf.global_variables_initializer()


def predict(tree_data):
    with tf.Session() as sess:
        sess.run(init)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        data_loader = BaseDataLoader(tree_data)
        test_batch_data = data_loader.make_minibatch_iterator()
            #print("***************")

            #print(test_batch_data["batch_labels"])
            # print(test_batch_data["batch_size"])
        scores = sess.run(
                [tbcnn_model.softmax],
                feed_dict={
                    tbcnn_model.placeholders["node_type"]: test_batch_data["batch_node_type_id"],
                    tbcnn_model.placeholders["node_token"]:  test_batch_data["batch_node_sub_tokens_id"],
                    tbcnn_model.placeholders["children_index"]:  test_batch_data["batch_children_index"],
                    tbcnn_model.placeholders["children_node_type"]: test_batch_data["batch_children_node_type_id"],
                    tbcnn_model.placeholders["children_node_token"]: test_batch_data["batch_children_node_sub_tokens_id"],
                    tbcnn_model.placeholders["labels"]: test_batch_data["batch_labels_one_hot"],
                    tbcnn_model.placeholders["dropout_rate"]: 0.0
                }
        )
        prediction = np.argmax(scores[0],axis=1)[0]
        #label = test_batch_data["batch_labels"]
        #print(prediction)
        #print(label)
    return prediction


class BaseDataLoader():
   
    def __init__(self, tree_data):
        self.buckets = tree_data
        self.label_size = 104
        self.tree_size_threshold_upper = 0
        self.tree_size_threshold_lower = 1500
        # self.make_minibatch_iterator()
    
    def _onehot(self, i, total):
        zeros = np.zeros(total)
        zeros[i] = 1.0
        return zeros

    def make_batch(self, tree_data):
        batch_node_index = []
        batch_node_type_id = []
        batch_node_sub_tokens_id = []
        batch_node_token = []

        batch_children_index = []
        batch_children_node_type_id = []
        batch_children_node_sub_tokens_id = []
        batch_children_node_token = []

        batch_labels = []
        batch_labels_one_hot = []
        batch_size = []

        tree_data = tree_data[0]

        batch_node_index.append(tree_data["node_index"])
        batch_node_type_id.append(tree_data["node_type_id"])
        batch_node_sub_tokens_id.append(tree_data["node_sub_tokens_id"])
        batch_node_token.append(tree_data["node_token"])

        batch_children_index.append(tree_data["children_index"])
        batch_children_node_type_id.append(tree_data["children_node_type_id"])
        batch_children_node_sub_tokens_id.append(tree_data["children_node_sub_tokens_id"])
        batch_children_node_token.append(tree_data["children_node_token"])

        batch_labels.append(tree_data["label"])
        batch_labels_one_hot.append(self._onehot(tree_data["label"], self.label_size))

        batch_size.append(tree_data["size"])
        
        # [[]]
        batch_node_index = self._pad_batch_2D(batch_node_index)
        # [[]]
        batch_node_type_id = self._pad_batch_2D(batch_node_type_id)
        # [[[]]]
        batch_node_sub_tokens_id = self._pad_batch_3D(batch_node_sub_tokens_id)
        # [[[]]]
        batch_children_index = self._pad_batch_3D(batch_children_index)
        # [[[]]]
        batch_children_node_type_id = self._pad_batch_3D(batch_children_node_type_id)    
        # [[[[]]]]
        batch_children_node_sub_tokens_id = self._pad_batch_4D(batch_children_node_sub_tokens_id)

        batch_obj = {
            "batch_node_index": batch_node_index,
            "batch_node_type_id": batch_node_type_id,
            "batch_node_sub_tokens_id": batch_node_sub_tokens_id,
            "batch_children_index": batch_children_index,
            "batch_children_node_type_id": batch_children_node_type_id,
            "batch_children_node_sub_tokens_id": batch_children_node_sub_tokens_id,
            "batch_labels": batch_labels,
            "batch_labels_one_hot": batch_labels_one_hot,
            "batch_size": batch_size
        }
        return batch_obj


    def _pad_batch_2D(self, batch):
        max_batch = max([len(x) for x in batch])
        batch = [n + [0] * (max_batch - len(n)) for n in batch]
        batch = np.asarray(batch)
        return batch

    def _pad_batch_3D(self, batch):
        max_2nd_D = max([len(x) for x in batch])
        max_3rd_D = max([len(c) for n in batch for c in n])
        batch = [n + ([[]] * (max_2nd_D - len(n))) for n in batch]
        batch = [[c + [0] * (max_3rd_D - len(c)) for c in sample] for sample in batch]
        batch = np.asarray(batch)
        return batch


    def _pad_batch_4D(self, batch):
        max_2nd_D = max([len(x) for x in batch])
        max_3rd_D = max([len(c) for n in batch for c in n])
        max_4th_D = max([len(s) for n in batch for c in n for s in c])
        batch = [n + ([[]] * (max_2nd_D - len(n))) for n in batch]
        batch = [[c + ([[]] * (max_3rd_D - len(c))) for c in sample] for sample in batch]
        batch = [[[s + [0] * (max_4th_D - len(s)) for s in c] for c in sample] for sample in batch]
        batch = np.asarray(batch)
        return batch

    def make_minibatch_iterator(self):
        elements = []
        elements.append(self.buckets)
        batch_obj = self.make_batch(elements)                
        return batch_obj


#load original data
original_data_path = '../OJ104_pycparser_train_test_val/pycparser-buckets-train.pkl'
dif_root = '../dif_file'
node_type_vocab_path = '../vocab/pycparser/node_type/type.txt'
node_token_vocab_path = '../vocab/pycparser/node_token/token.txt'
dif_num = 5

dif_buckets = defaultdict(list)

buckets = pickle.load(open(original_data_path, "rb" ))
bucket_ids = list(buckets.keys())
log_file = open("log_dif.txt", 'a')
print("length: ",len(bucket_ids), file=log_file)
log_file.close()
for bucket_idx in bucket_ids:
    log_file = open("log_dif.txt", 'a')
    print("bucket_id: ", bucket_idx, file=log_file)
    log_file.close()
    bucket_data = buckets[bucket_idx]
    for i, ele in enumerate(bucket_data):
        log_file = open("log_dif.txt", 'a')
        print(ele['file_path'], file=log_file)
        log_file.close()
        ori_tree_data = ele
        ori_label = predict(ori_tree_data)
        dif_path = os.path.join(dif_root, ele['file_path'].split('/')[-3], ele['file_path'].split('/')[-2], ele['file_path'].split('/')[-1].split('.')[0])
        if not os.path.exists(dif_path):
            os.mkdir(dif_path)
        count = 0
        for dif_file in os.listdir(dif_path):
            if count >= dif_num:
                break
            dif_file_path = os.path.join(dif_path, dif_file)
            print(dif_file_path)
            dif_data_processor = PycParserDataProcessor(node_type_vocab_path, node_token_vocab_path, dif_file_path, 'pycparser')
            dif_tree_data = dif_data_processor.tree_data
            if dif_tree_data is None:
                continue
            dif_pre = predict(dif_tree_data)
            if dif_pre == ori_label:
                dif_buckets[bucket_idx].append(dif_tree_data)
                count += 1
        while count < dif_num:
            dif_buckets[bucket_idx].append(ori_tree_data)
            count += 1
pickle.dump(dif_buckets, open("../dif_file/dif-buckets-train.pkl", "wb" ) )    
