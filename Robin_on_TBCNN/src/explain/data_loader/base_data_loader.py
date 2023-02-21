import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random

class BaseDataLoader():
   
    def __init__(self, batch_size, label_size, tree_size_threshold_upper, tree_size_threshold_lower, data_path, dif_data_path, is_training=True):

        self.is_training = is_training
        self.buckets = pickle.load(open(data_path, "rb" ))
        self.dif_buckets = pickle.load(open(dif_data_path, "rb"))
        self.batch_size = batch_size
        self.label_size = label_size
        self.tree_size_threshold_upper = tree_size_threshold_upper
        self.tree_size_threshold_lower = tree_size_threshold_lower
        self.dif_num = 5
        # self.make_minibatch_iterator()
    
    def _onehot(self, i, total):
        zeros = np.zeros(total)
        zeros[i] = 1.0
        return zeros

    def make_batch(self, batch_data, dif_batch_data):
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

        dif_batch_node_index = []
        dif_batch_node_type_id = []
        dif_batch_node_sub_tokens_id = []

        dif_batch_children_index = []
        dif_batch_children_node_type_id = []
        dif_batch_children_node_sub_tokens_id = []

        for i, tree_data in enumerate(batch_data):
            #orginal data
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
            #dif data
            dif_batch_node_index.extend([d["node_index"] for d in dif_batch_data[i]])
            dif_batch_node_type_id.extend([d["node_type_id"] for d in dif_batch_data[i]])
            dif_batch_node_sub_tokens_id.extend([d["node_sub_tokens_id"] for d in dif_batch_data[i]])

            dif_batch_children_index.extend([d["children_index"] for d in dif_batch_data[i]])
            dif_batch_children_node_type_id.extend([d["children_node_type_id"] for d in dif_batch_data[i]])
            dif_batch_children_node_sub_tokens_id.extend([d["children_node_sub_tokens_id"] for d in dif_batch_data[i]])
        
        each_batch = len(batch_node_sub_tokens_id)
        # [[]]
        batch_node_index, dif_batch_node_index= self._pad_batch_2D(batch_node_index, dif_batch_node_index)
        dif_batch_node_index = dif_batch_node_index.reshape(each_batch, self.dif_num, -1)
        # [[]]
        batch_node_type_id, dif_batch_node_type_id = self._pad_batch_2D(batch_node_type_id, dif_batch_node_type_id)
        dif_batch_node_type_id = dif_batch_node_type_id.reshape(each_batch, self.dif_num, -1)
        # [[[]]]
        batch_node_sub_tokens_id, dif_batch_node_sub_tokens_id = self._pad_batch_3D(batch_node_sub_tokens_id, dif_batch_node_sub_tokens_id)
        dif_batch_node_sub_tokens_id = dif_batch_node_sub_tokens_id.reshape(each_batch, self.dif_num, dif_batch_node_sub_tokens_id.shape[1], dif_batch_node_sub_tokens_id.shape[2])
        # [[[]]]
        batch_children_index, dif_batch_children_index = self._pad_batch_3D(batch_children_index, dif_batch_children_index)
        dif_batch_children_index = dif_batch_children_index.reshape(each_batch, self.dif_num, dif_batch_children_index.shape[1], dif_batch_children_index.shape[2])
        # [[[]]]
        batch_children_node_type_id, dif_batch_children_node_type_id = self._pad_batch_3D(batch_children_node_type_id, dif_batch_children_node_type_id)
        dif_batch_children_node_type_id = dif_batch_children_node_type_id.reshape(each_batch, self.dif_num, dif_batch_children_node_type_id.shape[1], dif_batch_children_node_type_id.shape[2])
        # [[[[]]]]
        batch_children_node_sub_tokens_id, dif_batch_children_node_sub_tokens_id = self._pad_batch_4D(batch_children_node_sub_tokens_id, dif_batch_children_node_sub_tokens_id)
        dif_batch_children_node_sub_tokens_id = dif_batch_children_node_sub_tokens_id.reshape(each_batch, self.dif_num, dif_batch_children_node_sub_tokens_id.shape[1], dif_batch_children_node_sub_tokens_id.shape[2], dif_batch_children_node_sub_tokens_id.shape[3])

        batch_obj = {
            "batch_node_index": batch_node_index,
            "batch_node_type_id": batch_node_type_id,
            "batch_node_sub_tokens_id": batch_node_sub_tokens_id,
            "batch_children_index": batch_children_index,
            "batch_children_node_type_id": batch_children_node_type_id,
            "batch_children_node_sub_tokens_id": batch_children_node_sub_tokens_id,
            "batch_labels": batch_labels,
            "batch_labels_one_hot": batch_labels_one_hot,
            "batch_size": batch_size,
            # [batch_size, dif_num, ..]
            "dif_batch_node_index": dif_batch_node_index,
            "dif_batch_node_type_id": dif_batch_node_type_id,
            "dif_batch_node_sub_tokens_id": dif_batch_node_sub_tokens_id,
            "dif_batch_children_index": dif_batch_children_index,
            "dif_batch_children_node_type_id": dif_batch_children_node_type_id,
            "dif_batch_children_node_sub_tokens_id": dif_batch_children_node_sub_tokens_id
        }
        return batch_obj


    def _pad_batch_2D(self, batch, dif_batch):
        max_batch = max(max([len(x) for x in batch]), max([len(x) for x in dif_batch]))
        batch = [n + [0] * (max_batch - len(n)) for n in batch]
        batch = np.asarray(batch)
        dif_batch = [n + [0] * (max_batch - len(n)) for n in dif_batch]
        dif_batch = np.asarray(dif_batch)
        return batch, dif_batch

    def _pad_batch_3D(self, batch, dif_batch):
        max_2nd_D = max(max([len(x) for x in batch]), max([len(x) for x in dif_batch]))
        max_3rd_D = max(max([len(c) for n in batch for c in n]), max([len(c) for n in dif_batch for c in n]))
        batch = [n + ([[]] * (max_2nd_D - len(n))) for n in batch]
        batch = [[c + [0] * (max_3rd_D - len(c)) for c in sample] for sample in batch]
        batch = np.asarray(batch)
        dif_batch = [n + ([[]] * (max_2nd_D - len(n))) for n in dif_batch]
        dif_batch = [[c + [0] * (max_3rd_D - len(c)) for c in sample] for sample in dif_batch]
        dif_batch = np.asarray(dif_batch)
        return batch, dif_batch


    def _pad_batch_4D(self, batch, dif_batch):
        max_2nd_D = max(max([len(x) for x in batch]), max([len(x) for x in dif_batch]))
        max_3rd_D = max(max([len(c) for n in batch for c in n]), max([len(c) for n in dif_batch for c in n]))
        max_4th_D = max(max([len(s) for n in batch for c in n for s in c]), max([len(s) for n in dif_batch for c in n for s in c]))
        batch = [n + ([[]] * (max_2nd_D - len(n))) for n in batch]
        batch = [[c + ([[]] * (max_3rd_D - len(c))) for c in sample] for sample in batch]
        batch = [[[s + [0] * (max_4th_D - len(s)) for s in c] for c in sample] for sample in batch]
        batch = np.asarray(batch)
        dif_batch = [n + ([[]] * (max_2nd_D - len(n))) for n in dif_batch]
        dif_batch = [[c + ([[]] * (max_3rd_D - len(c))) for c in sample] for sample in dif_batch]
        dif_batch = [[[s + [0] * (max_4th_D - len(s)) for s in c] for c in sample] for sample in dif_batch]
        dif_batch = np.asarray(dif_batch)
        return batch, dif_batch

    def make_minibatch_iterator(self):

        bucket_ids = list(self.buckets.keys())
        random.shuffle(bucket_ids)
        
        
        for bucket_idx in bucket_ids:

            bucket_data = self.buckets[bucket_idx]
            dif_bucket_data = self.dif_buckets[bucket_idx]
            # reshape
            dif_bucket_data_re = []
            for i in range(0, len(dif_bucket_data), self.dif_num):
                dif_bucket_data_re.append(dif_bucket_data[i:i+self.dif_num])
            randnum = np.random.randint(0, 1234)
            np.random.seed(randnum)
            np.random.shuffle(bucket_data)
            np.random.seed(randnum)
            np.random.shuffle(dif_bucket_data_re)
            #random.shuffle(bucket_data)
            
            elements = []
            dif_elements = []
            samples = 0

            for i, ele in enumerate(bucket_data):
                if self.is_training == True:
                    if ele["size"] > self.tree_size_threshold_lower and ele["size"] < self.tree_size_threshold_upper:
                        elements.append(ele)
                        dif_elements.append(dif_bucket_data_re[i])
                        samples += 1
                else:
                    elements.append(ele)
                    dif_elements.append(dif_bucket_data_re[i])
                    samples += 1

              
                if samples >= self.batch_size:
                    batch_obj = self.make_batch(elements, dif_elements)       
                    yield batch_obj
                    elements = []
                    dif_elements = []
                    samples = 0
        