import math
import tensorflow as tf

from base_layer import BaseLayer
import numpy as np
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()


class TBCNN_explainer(BaseLayer):
    def __init__(self, opt):
        super().__init__(opt)
        self.node_init = opt.node_init
        self.num_conv = opt.num_conv

        self.conv_output_dim = opt.conv_output_dim
        self.node_token_dim = opt.node_token_dim
        self.node_type_dim = opt.node_type_dim
        self.node_dim = self.conv_output_dim
        self.training_loss_option = opt.training_loss_option
        self.dif_num = 5
        self.init_tbcnn_params()
       
    def init_tbcnn_params(self):

        """Initialize parameters"""
           
        self.placeholders["node_type"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None), name='node_type')
        self.placeholders["node_token"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None, None), name='node_token')
        self.placeholders["children_node_token"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None, None, None), name='children_token') # batch_size x max_num_nodes x max_children x max_sub_tokens

        self.placeholders["children_index"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None, None), name='children_index') # batch_size x max_num_nodes x max_children
        self.placeholders["children_node_type"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None, None), name='children_type') # batch_size x max_num_nodes x max_children
            
        self.placeholders["labels"] = tf.compat.v1.placeholder(tf.float32, shape=(None, None), name="labels")

        self.placeholders["dropout_rate"] = tf.compat.v1.placeholder(tf.float32)

        self.placeholders["dif_node_type"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None, None), name='node_type')
        self.placeholders["dif_node_token"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None, None, None), name='node_token')
        self.placeholders["dif_children_node_token"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None, None, None, None), name='children_token') # batch_size x max_num_nodes x max_children x max_sub_tokens

        self.placeholders["dif_children_index"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None, None, None), name='children_index') # batch_size x max_num_nodes x max_children
        self.placeholders["dif_children_node_type"] = tf.compat.v1.placeholder(tf.int32, shape=(None, None, None, None), name='children_type') # batch_size x max_num_nodes x max_children
            

        with tf.compat.v1.name_scope('approximator_params'):
            self.weights_sapp = {}
            self.weights_usapp = {}
            # select_approximator_model
            for i in range(self.num_conv):
                self.weights_sapp["w_t_" + str(i)] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, self.conv_output_dim]), name='w_t_' + str(i))
                self.weights_sapp["w_l_" + str(i)] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, self.conv_output_dim]), name='w_l_' + str(i))
                self.weights_sapp["w_r_" + str(i)] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, self.conv_output_dim]), name='w_r_' + str(i))
                self.weights_sapp["b_conv_" + str(i)] = tf.Variable(tf.zeros([self.conv_output_dim,]),name='b_conv_' + str(i))

            self.weights_sapp["w_attention"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, 1]), name="w_attention")
            
            if self.node_init == 0:
                print("Using only type weights..........")
                self.weights_sapp["node_type_embedding"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([len(self.node_type_lookup.keys()), self.node_type_dim]), name='node_type_embedding')
            else:
                if self.node_init == 1:
                    print("Using only token weights..........")            
                    self.weights_sapp["node_token_embedding"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([len(self.node_token_lookup.keys()), self.node_token_dim]), name='node_token_embedding')
                else:
                    print("Using both type and token weights..........") 
                    self.weights_sapp["node_token_embedding"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([len(self.node_token_lookup.keys()), self.node_token_dim]), name='node_token_embedding')
                    self.weights_sapp["node_type_embedding"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([len(self.node_type_lookup.keys()), self.node_type_dim]), name='node_type_embedding')
            
            # unselect_approximator_model
            for i in range(self.num_conv):
                self.weights_usapp["w_t_" + str(i)] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, self.conv_output_dim]), name='w_t_' + str(i))
                self.weights_usapp["w_l_" + str(i)] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, self.conv_output_dim]), name='w_l_' + str(i))
                self.weights_usapp["w_r_" + str(i)] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, self.conv_output_dim]), name='w_r_' + str(i))
                self.weights_usapp["b_conv_" + str(i)] = tf.Variable(tf.zeros([self.conv_output_dim,]),name='b_conv_' + str(i))

            self.weights_usapp["w_attention"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, 1]), name="w_attention")
            
            if self.node_init == 0:
                print("Using only type weights..........")
                self.weights_usapp["node_type_embedding"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([len(self.node_type_lookup.keys()), self.node_type_dim]), name='node_type_embedding')
            else:
                if self.node_init == 1:
                    print("Using only token weights..........")            
                    self.weights_usapp["node_token_embedding"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([len(self.node_token_lookup.keys()), self.node_token_dim]), name='node_token_embedding')
                else:
                    print("Using both type and token weights..........") 
                    self.weights_usapp["node_token_embedding"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([len(self.node_token_lookup.keys()), self.node_token_dim]), name='node_token_embedding')
                    self.weights_usapp["node_type_embedding"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([len(self.node_type_lookup.keys()), self.node_type_dim]), name='node_type_embedding')

        with tf.compat.v1.name_scope('explainer_params'):
            self.weights_exp = {}
            #explainer_model
            for i in range(self.num_conv):
                self.weights_exp["w_t_" + str(i)] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, self.conv_output_dim]), name='w_t_' + str(i))
                self.weights_exp["w_l_" + str(i)] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, self.conv_output_dim]), name='w_l_' + str(i))
                self.weights_exp["w_r_" + str(i)] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, self.conv_output_dim]), name='w_r_' + str(i))
                self.weights_exp["b_conv_" + str(i)] = tf.Variable(tf.zeros([self.conv_output_dim,]),name='b_conv_' + str(i))

            self.weights_exp["w_attention"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([self.node_dim, 1]), name="w_attention")
            
            print("Using both type and token weights..........") 
            self.weights_exp["node_token_embedding"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([len(self.node_token_lookup.keys()), self.node_token_dim]), name='node_token_embedding')
            self.weights_exp["node_type_embedding"] = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")([len(self.node_type_lookup.keys()), self.node_type_dim]), name='node_type_embedding')
        
    def Cal_dif(self):
        node_type = tf.cast(self.placeholders["node_type"], tf.float32)
        bsize = tf.shape(node_type)[0]
        nodenum = tf.shape(node_type)[1]
        smask = tf.reshape(tf.tile(self.mask, [1, self.dif_num]), [bsize * self.dif_num, nodenum])
        dif_mask = self.dif_mask
        same_num = tf.compat.v1.count_nonzero(tf.multiply(smask, dif_mask), axis=-1, dtype=tf.float32)
        avg_sum = tf.reduce_mean(same_num, axis=-1)
        return avg_sum

    def shuffle_dif(self):
        dif_node_type = tf.split(self.placeholders["dif_node_type"], self.dif_num, axis=1)[0]
        dif_children_index = tf.split(self.placeholders["dif_children_index"], self.dif_num, axis=1)[0]
        dif_node_token = tf.split(self.placeholders["dif_node_token"], self.dif_num, axis=1)[0]
        dif_children_node_token = tf.split(self.placeholders["dif_children_node_token"], self.dif_num, axis=1)[0]
        dif_labels = self.placeholders["labels"]
        index = [i for i in range(32)]
        np.random.shuffle(index)
        node_type = dif_node_type[index[0]]
        for i in index[1:]:
            node_type = tf.concat([node_type, dif_node_type[i]], axis=0)
        children_index = dif_children_index[index[0]]
        for i in index[1:]:
            children_index = tf.concat([children_index, dif_children_index[i]], axis=0)
        node_token = dif_node_token[index[0]]
        for i in index[1:]:
            node_token = tf.concat([node_token, dif_node_token[i]], axis=0)
        children_node_token = dif_children_node_token[index[0]]
        for i in index[1:]:
            children_node_token = tf.concat([children_node_token, dif_children_node_token[i]], axis=0)    
        labels = tf.expand_dims(dif_labels[index[0]],0)
        for i in index[1:]:
            labels = tf.concat([labels, tf.expand_dims(dif_labels[i],0)], axis=0)    
        '''
        print(node_type.shape)
        print(children_index.shape)
        print(node_token.shape)
        print(children_node_token.shape)
        print(labels.shape)
        '''
        return node_type, children_index, node_token, children_node_token, labels


    def explainer(self):
        #### selection
        _, self.sel_last_conv_output = self.convolve_tree_based_cnn_exp(self.placeholders["node_type"], self.placeholders["children_index"], self.placeholders["node_token"], self.placeholders["children_node_token"])
        self.code_logit = self.sel_aggregation_layer(self.sel_last_conv_output)
        
        self.tau0 = 0.5 
        self.k = 10
        self.mask = self.Sample_Concrete(self.code_logit)
        self.umask = tf.subtract(1., self.mask)

        #reshape
        dif_node_type = tf.reshape(self.placeholders["dif_node_type"], [tf.shape(self.placeholders["dif_node_type"])[0] * tf.shape(self.placeholders["dif_node_type"])[1], tf.shape(self.placeholders["dif_node_type"])[2]])
        dif_children_index = tf.reshape(self.placeholders["dif_children_index"], [tf.shape(self.placeholders["dif_children_index"])[0] * tf.shape(self.placeholders["dif_children_index"])[1], tf.shape(self.placeholders["dif_children_index"])[2], tf.shape(self.placeholders["dif_children_index"])[3]])
        dif_node_token = tf.reshape(self.placeholders["dif_node_token"], [tf.shape(self.placeholders["dif_node_token"])[0] * tf.shape(self.placeholders["dif_node_token"])[1], tf.shape(self.placeholders["dif_node_token"])[2], tf.shape(self.placeholders["dif_node_token"])[3]])
        dif_children_node_token = tf.reshape(self.placeholders["dif_children_node_token"], [tf.shape(self.placeholders["dif_children_node_token"])[0] * tf.shape(self.placeholders["dif_children_node_token"])[1], tf.shape(self.placeholders["dif_children_node_token"])[2], tf.shape(self.placeholders["dif_children_node_token"])[3], tf.shape(self.placeholders["dif_children_node_token"])[4]])
        
        _, self.dif_sel_last_conv_output = self.convolve_tree_based_cnn_exp(dif_node_type, dif_children_index, dif_node_token, dif_children_node_token)
        self.dif_code_logit = self.sel_aggregation_layer(self.dif_sel_last_conv_output)
        self.dif_mask = self.Sample_Concrete(self.dif_code_logit)

        self.dif_loss = self.Cal_dif()

        self.shuffle_node_type, self.shuffle_children_index, self.shuffle_node_token, self.shuffle_children_node_token, shuffle_labels = self.shuffle_dif()
        #print(self.placeholders["node_type"].shape, self.placeholders["children_index"].shape, self.placeholders["node_token"].shape, self.placeholders["children_node_token"].shape)
        #print(self.shuffle_node_type.shape, self.shuffle_children_index.shape, self.shuffle_node_token.shape, self.shuffle_children_node_token.shape)
        _, self.mix_conv_output = self.convolve_tree_based_cnn_exp(self.shuffle_node_type, self.shuffle_children_index, self.shuffle_node_token, self.shuffle_children_node_token)
        self.mix_labels = shuffle_labels
        self.lams = np.random.beta(0.05, 0.1, 32)
        self.mix_conv_output_new=tf.expand_dims((1-self.lams[0]) *self.sel_last_conv_output[0] + self.lams[0] * self.mix_conv_output[0], 0)
        self.mix_labels_new=tf.expand_dims((1-self.lams[0]) *self.placeholders["labels"][0] + self.lams[0] * self.mix_labels[0], 0)
        for i in range(1, 32):
            self.mix_conv_outputi=(1-self.lams[i]) *self.sel_last_conv_output[i] + self.lams[i] * self.mix_conv_output[i]
            self.mix_labelsi=(1-self.lams[i]) *self.placeholders["labels"][i] + self.lams[i] * self.mix_labels[i]
            self.mix_conv_output_new = tf.concat([self.mix_conv_output_new,tf.expand_dims(self.mix_conv_outputi,0)],axis=0)
            self.mix_labels_new = tf.concat([self.mix_labels_new,tf.expand_dims(self.mix_labelsi,0)],axis=0)
        self.mix_conv_output = self.mix_conv_output_new
        self.mix_labels = self.mix_labels_new
        self.mix_code_logit = self.sel_aggregation_layer(self.mix_conv_output)
        self.mix_mask = self.Sample_Concrete(self.mix_code_logit)
        self.mix_umask = tf.subtract(1., self.mask)     
        

    def approximator(self):
        
        #### As
        _, self.s_conv_output = self.convolve_tree_based_cnn_sapp()
        self.s_code_vector = self.aggregation_layer(self.s_conv_output)
        with tf.compat.v1.variable_scope("approximator_2", reuse=tf.compat.v1.AUTO_REUSE):
            self.s_logits = tf.compat.v1.layers.dense(self.s_code_vector, self.label_size, activation=tf.nn.leaky_relu, name='approximator_dense1')
        
        _, self.mix_s_conv_output = self.mix_convolve_tree_based_cnn_sapp()
        self.mix_s_code_vector = self.aggregation_layer(self.mix_s_conv_output)
        with tf.compat.v1.variable_scope("approximator_2", reuse=tf.compat.v1.AUTO_REUSE):
            self.mix_s_logits = tf.compat.v1.layers.dense(self.mix_s_code_vector, self.label_size, activation=tf.nn.leaky_relu, name='approximator_dense1')
        

        #### Au
        _, self.us_conv_output = self.convolve_tree_based_cnn_usapp()
        self.us_code_vector = self.aggregation_layer(self.us_conv_output)
        self.us_logits = tf.compat.v1.layers.dense(self.us_code_vector, self.label_size, activation=tf.nn.leaky_relu, name='approximator_dense2')

        #### loss
        self.s_softmax = tf.nn.softmax(self.s_logits)
        self.s_softmax_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.s_logits, labels=self.placeholders["labels"])
        self.s_loss = tf.reduce_mean(input_tensor=self.s_softmax_loss)
        
        self.us_softmax = tf.nn.softmax(self.us_logits)
        self.us_softmax_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.us_logits, labels=self.placeholders["labels"])
        self.us_loss = tf.reduce_mean(input_tensor=self.us_softmax_loss)
       
        self.mix_s_softmax = tf.nn.softmax(self.mix_s_logits)
        self.mix_s_softmax_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.mix_s_logits, labels=self.mix_labels)
        self.mix_s_loss = tf.reduce_mean(input_tensor=self.mix_s_softmax_loss)
        


            

    def feed_forward(self):
        self.explainer()
        self.approximator()
        
        self.var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        self.pos_var_list = []
        self.neg_var_list = []
        for var in self.var_list:
            if 'approximator' in var.name:
                self.pos_var_list.append(var)
            elif 'explainer' in var.name:
                self.neg_var_list.append(var)
            else:
                assert(1==0)
        
        # train approximator
        self.pos_loss = self.s_loss + 3e-3 * self.us_loss + 1e-1 * self.mix_s_loss
        #self.pos_var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='approximator_params')
        
        # train explainer
        self.neg_loss = self.s_loss - 3e-3 * self.us_loss
        self.neg_loss_dif = self.s_loss - 3e-3 * self.us_loss + 1e-4 * self.dif_loss + 1e-1 * self.mix_s_loss
        #self.neg_var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='explainer_params')
               


    def convolve_tree_based_cnn_exp(self, node_type, children_index, node_token, children_node_token):
        # explainer
        
        print("Including both type and token information..........")

        parent_node_type_embedding = self.compute_parent_node_types_tensor(node_type, self.weights_exp["node_type_embedding"])
        # shape = (batch_size, max_tree_size, max_children, node_type_dim)
        # Example with batch size = 12: shape = (12, 48, 8, 30)
        children_node_type_embedding = self.compute_children_node_types_tensor(parent_node_type_embedding, children_index, self.node_type_dim)
        # shape = (batch_size, max_tree_size, node_token_dim)
        # Example with batch size = 12: shape = (12, 48, 50))
        parent_node_token_embedding = self.compute_parent_node_tokens_tensor(node_token, self.weights_exp["node_token_embedding"])
                    
        # shape = (batch_size, max_tree_size, max_children, node_token_dim)
        # Example with batch size = 12: shape = (12, 48, 7, 50)
        children_node_token_embedding = self.compute_children_node_tokens_tensor(children_node_token, self.node_token_dim, self.weights_exp["node_token_embedding"])
                
        # shape = (batch_size, max_tree_size, (node_type_dim + node_token_dim))
        # Example with batch size = 12: shape = (12, 48, (30 + 50))) = (12, 48, 80)
        
        with tf.compat.v1.variable_scope("explainer_2", reuse=tf.compat.v1.AUTO_REUSE):
            parent_node_embedding = tf.concat([parent_node_type_embedding, parent_node_token_embedding], -1)
            parent_node_embedding = tf.compat.v1.layers.dense(parent_node_embedding, units=self.node_dim, activation=tf.nn.leaky_relu, name='explainer_dense1')
            # shape = (batch_size, max_tree_size, max_children, (node_type_dim + node_token_dim))
            # Example with batch size = 12: shape = (12, 48, 7, (30 + 50))) = (12, 48, 6, 80)
            children_embedding = tf.concat([children_node_type_embedding, children_node_token_embedding], -1)
            children_embedding = tf.compat.v1.layers.dense(children_embedding, units=self.node_dim, activation=tf.nn.leaky_relu, name='explainer_dense2')
        
        """Tree based Convolutional Layer"""
        # Example with batch size = 12 and num_conv = 8: shape = (12, 48, 128, 8)
        conv_output = self.conv_layer_exp(parent_node_embedding, children_embedding, children_index, self.num_conv, self.node_dim)

        return conv_output

    def mix_convolve_tree_based_cnn_sapp(self):    
        # select_approximator
        basize = tf.shape(self.shuffle_node_type)[0]
        nodenum = tf.shape(self.shuffle_node_type)[1]
        smask = tf.reshape(self.mix_mask, [-1,1])
        mix_mask_parent_node_type_embedding = self.compute_parent_node_types_tensor(self.shuffle_node_type, self.weights_sapp["node_type_embedding"])
        node_type_shape = [basize * nodenum, -1]
        mix_node_type = tf.reshape(mix_mask_parent_node_type_embedding, node_type_shape)
        mix_node_type_smask = tf.tile(smask, multiples=[1,tf.shape(mix_node_type)[1]])
        mix_mask_parent_node_type_embedding = tf.reshape(self.mul_grad(mix_node_type, mix_node_type_smask), tf.shape(mix_mask_parent_node_type_embedding))
        mix_mask_children_node_type_embedding = self.compute_children_node_types_tensor(mix_mask_parent_node_type_embedding, self.shuffle_children_index, self.node_type_dim)
        mix_mask_parent_node_token_embedding = self.compute_parent_node_tokens_tensor(self.shuffle_node_token, self.weights_sapp["node_token_embedding"])
        mix_mask_children_node_token_embedding = self.compute_children_node_tokens_tensor(self.shuffle_children_node_token, self.node_token_dim, self.weights_sapp["node_token_embedding"])
        mix_mask_parent_node_embedding = tf.concat([mix_mask_parent_node_type_embedding, mix_mask_parent_node_token_embedding], -1)

        with tf.compat.v1.variable_scope("approximator_2", reuse=tf.compat.v1.AUTO_REUSE):
            mix_mask_parent_node_embedding = tf.compat.v1.layers.dense(mix_mask_parent_node_embedding, units=self.node_dim, activation=tf.nn.leaky_relu, name='approximator_dense3')
            mix_mask_children_embedding = tf.concat([mix_mask_children_node_type_embedding, mix_mask_children_node_token_embedding], -1)
            mix_mask_children_embedding = tf.compat.v1.layers.dense(mix_mask_children_embedding, units=self.node_dim, activation=tf.nn.leaky_relu, name='approximator_dense4')
        """Tree based Convolutional Layer"""
        mix_children_index_shape = [basize * nodenum, -1]
        mix_children_index = tf.reshape(self.shuffle_children_index, mix_children_index_shape)
        mix_children_index_smask = tf.tile(smask, multiples=[1,tf.shape(mix_children_index)[1]])
        mix_mask_children_index = tf.reshape(self.mul_grad(mix_children_index, mix_children_index_smask), tf.shape(self.shuffle_children_index))
        mix_mask_children_index = self.cast_grad(mix_mask_children_index)

        mix_conv_output = self.conv_layer_sapp(mix_mask_parent_node_embedding, mix_mask_children_embedding, mix_mask_children_index, self.num_conv, self.node_dim)
        mix_conv_output_new = tf.expand_dims((1-self.lams[0]) * self.s_conv_output[0] + self.lams[0] * mix_conv_output[1][0], 0)
        for i in range(1,32):
            mix_conv_outputi = (1-self.lams[i]) * self.s_conv_output[i] + self.lams[i] * mix_conv_output[1][i]
            mix_conv_output_new = tf.concat([mix_conv_output_new, tf.expand_dims(mix_conv_outputi,0)],axis=0)
        return mix_conv_output[0], mix_conv_output_new

    def convolve_tree_based_cnn_sapp(self):    
        # select_approximator
        basize = tf.shape(self.placeholders["node_type"])[0]
        nodenum = tf.shape(self.placeholders["node_type"])[1]
        smask = tf.reshape(self.mask, [-1,1])
        print("Including both type and token information..........")
        # embedding
        # node_type
        self.mask_parent_node_type_embedding = self.compute_parent_node_types_tensor(self.placeholders["node_type"], self.weights_sapp["node_type_embedding"])
        # node_type
        node_type_shape = [basize * nodenum, -1]
        node_type = tf.reshape(self.mask_parent_node_type_embedding, node_type_shape)
        node_type_smask = tf.tile(smask, multiples=[1,tf.shape(node_type)[1]])
        self.mask_parent_node_type_embedding = tf.reshape(self.mul_grad(node_type, node_type_smask), tf.shape(self.mask_parent_node_type_embedding))
        #self.mask_parent_node_type_embedding = self.cast_grad(self.mask_parent_node_type_embedding)

        # children_node_type
        #self.mask_children_node_type_embedding = self.compute_children_node_types_tensor(self.mask_parent_node_type_embedding, self.maskinput["children_index"], self.node_type_dim)
        self.mask_children_node_type_embedding = self.compute_children_node_types_tensor(self.mask_parent_node_type_embedding, self.placeholders["children_index"], self.node_type_dim)
        
        # parent_node_token
        #self.mask_parent_node_token_embedding = self.compute_parent_node_tokens_tensor(self.maskinput["node_token"], self.weights_sapp["node_token_embedding"])
        self.mask_parent_node_token_embedding = self.compute_parent_node_tokens_tensor(self.placeholders["node_token"], self.weights_sapp["node_token_embedding"])
        
        # children_node_token
        #self.mask_children_node_token_embedding = self.compute_children_node_tokens_tensor(self.maskinput["children_node_token"], self.node_token_dim, self.weights_sapp["node_token_embedding"])
        self.mask_children_node_token_embedding = self.compute_children_node_tokens_tensor(self.placeholders["children_node_token"], self.node_token_dim, self.weights_sapp["node_token_embedding"])
        
        ### mask
        self.mask_parent_node_embedding = tf.concat([self.mask_parent_node_type_embedding, self.mask_parent_node_token_embedding], -1)
        with tf.compat.v1.variable_scope("approximator_2", reuse=tf.compat.v1.AUTO_REUSE):
            self.mask_parent_node_embedding = tf.compat.v1.layers.dense(self.mask_parent_node_embedding, units=self.node_dim, activation=tf.nn.leaky_relu, name='approximator_dense3')
            # shape = (batch_size, max_tree_size, max_children, (node_type_dim + node_token_dim))
            # Example with batch size = 12: shape = (12, 48, 7, (30 + 50))) = (12, 48, 6, 80)
            self.mask_children_embedding = tf.concat([self.mask_children_node_type_embedding, self.mask_children_node_token_embedding], -1)
            self.mask_children_embedding = tf.compat.v1.layers.dense(self.mask_children_embedding, units=self.node_dim, activation=tf.nn.leaky_relu, name='approximator_dense4')
        """Tree based Convolutional Layer"""
        # Example with batch size = 12 and num_conv = 8: shape = (12, 48, 128, 8)
        # children_index
        children_index_shape = [basize * nodenum, -1]
        children_index = tf.reshape(self.placeholders["children_index"], children_index_shape)
        children_index_smask = tf.tile(smask, multiples=[1,tf.shape(children_index)[1]])
        self.mask_children_index = tf.reshape(self.mul_grad(children_index, children_index_smask), tf.shape(self.placeholders["children_index"]))
        self.mask_children_index = self.cast_grad(self.mask_children_index)

        conv_output = self.conv_layer_sapp(self.mask_parent_node_embedding, self.mask_children_embedding, self.mask_children_index, self.num_conv, self.node_dim)
        return conv_output


    def convolve_tree_based_cnn_usapp(self):
        # unselect_approximator
        basize = tf.shape(self.placeholders["node_type"])[0]
        nodenum = tf.shape(self.placeholders["node_type"])[1]
        sumask = tf.reshape(self.umask, [-1,1])
        print("Including both type and token information..........")
        # embedding
        # node_type
        self.umask_parent_node_type_embedding = self.compute_parent_node_types_tensor(self.placeholders["node_type"], self.weights_usapp["node_type_embedding"])
        # node_type
        node_type_shape = [basize * nodenum, -1]
        node_type = tf.reshape(self.umask_parent_node_type_embedding, node_type_shape)
        node_type_sumask = tf.tile(sumask, multiples=[1,tf.shape(node_type)[1]])
        self.umask_parent_node_type_embedding = tf.reshape(self.mul_grad(node_type, node_type_sumask), tf.shape(self.umask_parent_node_type_embedding))
        #self.umask_parent_node_type_embedding = self.cast_grad(self.umask_parent_node_type_embedding)


        # children_node_type
        #self.mask_children_node_type_embedding = self.compute_children_node_types_tensor(self.mask_parent_node_type_embedding, self.maskinput["children_index"], self.node_type_dim)
        self.umask_children_node_type_embedding = self.compute_children_node_types_tensor(self.umask_parent_node_type_embedding, self.placeholders["children_index"], self.node_type_dim)
        
        # parent_node_token
        #self.mask_parent_node_token_embedding = self.compute_parent_node_tokens_tensor(self.maskinput["node_token"], self.weights_usapp["node_token_embedding"])
        self.umask_parent_node_token_embedding = self.compute_parent_node_tokens_tensor(self.placeholders["node_token"], self.weights_usapp["node_token_embedding"])
        
        # children_node_token
        #self.mask_children_node_token_embedding = self.compute_children_node_tokens_tensor(self.maskinput["children_node_token"], self.node_token_dim, self.weights_usapp["node_token_embedding"])
        self.umask_children_node_token_embedding = self.compute_children_node_tokens_tensor(self.placeholders["children_node_token"], self.node_token_dim, self.weights_usapp["node_token_embedding"])
        
        ### mask
        
        self.umask_parent_node_embedding = tf.concat([self.umask_parent_node_type_embedding, self.umask_parent_node_token_embedding], -1)
        self.umask_parent_node_embedding = tf.compat.v1.layers.dense(self.umask_parent_node_embedding, units=self.node_dim, activation=tf.nn.leaky_relu, name='approximator_dense5')
        # shape = (batch_size, max_tree_size, max_children, (node_type_dim + node_token_dim))
        # Example with batch size = 12: shape = (12, 48, 7, (30 + 50))) = (12, 48, 6, 80)
        self.umask_children_embedding = tf.concat([self.umask_children_node_type_embedding, self.umask_children_node_token_embedding], -1)
        self.umask_children_embedding = tf.compat.v1.layers.dense(self.umask_children_embedding, units=self.node_dim, activation=tf.nn.leaky_relu, name='approximator_dense6')
        """Tree based Convolutional Layer"""
        # Example with batch size = 12 and num_conv = 8: shape = (12, 48, 128, 8)
        # children_index
        children_index_shape = [basize * nodenum, -1]
        children_index = tf.reshape(self.placeholders["children_index"], children_index_shape)
        children_index_sumask = tf.tile(sumask, multiples=[1,tf.shape(children_index)[1]])
        self.umask_children_index = tf.reshape(self.mul_grad(children_index, children_index_sumask), tf.shape(self.placeholders["children_index"]))
        self.umask_children_index = self.cast_grad(self.umask_children_index)
        conv_output = self.conv_layer_usapp(self.umask_parent_node_embedding, self.umask_children_embedding, self.umask_children_index, self.num_conv, self.node_dim)
        return conv_output


    def aggregation_layer(self, nodes_representation):
        return tf.reduce_max(nodes_representation, axis=1)

    
    def sel_aggregation_layer(self, nodes_representation):
        return tf.reduce_max(nodes_representation, axis=2, keepdims = True)
        #return nodes_representation
    

    
    def Sample_Concrete(self, logits):
        """
        Layer for sample Concrete / Gumbel-Softmax variables. 

        """
        # logits: [batch_size, d, 1]
        logits_ = tf.compat.v1.keras.backend.permute_dimensions(logits, (0,2,1))# [batch_size, 1, d]

        batch_size = tf.shape(input=logits_)[0]
        d = tf.shape(input=logits_)[2]
        unif_shape = (batch_size, self.k, d)

        uniform = tf.random.uniform(shape=unif_shape,
            minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
            maxval = 1.0)
        gumbel = - tf.compat.v1.log(-tf.compat.v1.log(uniform))
        noisy_logits = (gumbel + logits_)/self.tau0
        samples = tf.nn.softmax(noisy_logits)
        samples = tf.compat.v1.keras.backend.max(samples, axis = 1)

        logits = tf.reshape(logits,[-1, d]) 
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)
        
        output = tf.compat.v1.keras.backend.in_train_phase(samples, discrete_logits) 
        return output


    @tf.custom_gradient
    def mul_grad(self, x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        def grad(dy):
            dx = y
            dy = x
            return dx, dy
        output = tf.multiply(x, y)
        # output = tf.cast(output, tf.int32)
        return output, grad
    
    @tf.custom_gradient
    def cast_grad(self, x):
        def grad(dy):
            return tf.ones_like(x)
        return tf.cast(x, tf.int32), grad

    # def aggregation_layer(self, conv):
    #     # conv is (batch_size, max_tree_size, conv_output_dim)
    #     with tf.name_scope("global_attention"):
    #         batch_size = tf.shape(conv)[0]
    #         max_tree_size = tf.shape(conv)[1]

    #         contexts_sum = tf.reduce_sum(conv, axis=1)
    #         contexts_sum_average = tf.divide(contexts_sum, tf.to_float(tf.expand_dims(max_tree_size, -1)))
          
    #         return contexts_sum_average


    def conv_node_exp(self, parent_node_embeddings, children_embeddings, children_indices, node_dim, layer):
        """Perform convolutions over every batch sample."""
        with tf.compat.v1.name_scope('conv_node_exp'):
            w_t, w_l, w_r = self.weights_exp["w_t_" + str(layer)], self.weights_exp["w_l_" + str(layer)], self.weights_exp["w_r_" + str(layer)]
            b_conv = self.weights_exp["b_conv_" + str(layer)]
        return self.conv_step(parent_node_embeddings, children_embeddings, children_indices, node_dim, w_t, w_r, w_l, b_conv)

    def conv_node_sapp(self, parent_node_embeddings, children_embeddings, children_indices, node_dim, layer):
        """Perform convolutions over every batch sample."""
        with tf.compat.v1.name_scope('conv_node_sapp'):
            w_t, w_l, w_r = self.weights_sapp["w_t_" + str(layer)], self.weights_sapp["w_l_" + str(layer)], self.weights_sapp["w_r_" + str(layer)]
            b_conv = self.weights_sapp["b_conv_" + str(layer)]
        return self.conv_step(parent_node_embeddings, children_embeddings, children_indices, node_dim, w_t, w_r, w_l, b_conv)

    def conv_node_usapp(self, parent_node_embeddings, children_embeddings, children_indices, node_dim, layer):
        """Perform convolutions over every batch sample."""
        with tf.compat.v1.name_scope('conv_node_usapp'):
            w_t, w_l, w_r = self.weights_usapp["w_t_" + str(layer)], self.weights_usapp["w_l_" + str(layer)], self.weights_usapp["w_r_" + str(layer)]
            b_conv = self.weights_usapp["b_conv_" + str(layer)]
        return self.conv_step(parent_node_embeddings, children_embeddings, children_indices, node_dim, w_t, w_r, w_l, b_conv)

    def conv_layer_exp(self, parent_node_embedding, children_embedding, children_index, num_conv, node_dim):
        with tf.compat.v1.name_scope('conv_layer_exp'):
            nodes = []
            for layer in range(num_conv):
                parent_node_embedding = self.conv_node_exp(parent_node_embedding, children_embedding, children_index, node_dim, layer)
                children_embedding = self.compute_children_node_types_tensor(parent_node_embedding, children_index, node_dim)
                nodes.append(tf.expand_dims(parent_node_embedding, axis=-1))
                # nodes = tf.expand_dims(parent_node_embeddings, axis=-1)
            return nodes, parent_node_embedding
    
    def conv_layer_sapp(self, parent_node_embedding, children_embedding, children_index, num_conv, node_dim):
        with tf.compat.v1.name_scope('conv_layer_sapp'):
            nodes = []
            for layer in range(num_conv):
                parent_node_embedding = self.conv_node_sapp(parent_node_embedding, children_embedding, children_index, node_dim, layer)
                children_embedding = self.compute_children_node_types_tensor(parent_node_embedding, children_index, node_dim)
                nodes.append(tf.expand_dims(parent_node_embedding, axis=-1))
                # nodes = tf.expand_dims(parent_node_embeddings, axis=-1)
            return nodes, parent_node_embedding
    
    def conv_layer_usapp(self, parent_node_embedding, children_embedding, children_index, num_conv, node_dim):
        with tf.compat.v1.name_scope('conv_layer_usapp'):
            nodes = []
            for layer in range(num_conv):
                parent_node_embedding = self.conv_node_usapp(parent_node_embedding, children_embedding, children_index, node_dim, layer)
                children_embedding = self.compute_children_node_types_tensor(parent_node_embedding, children_index, node_dim)
                nodes.append(tf.expand_dims(parent_node_embedding, axis=-1))
                # nodes = tf.expand_dims(parent_node_embeddings, axis=-1)
            return nodes, parent_node_embedding

    def conv_step(self, parent_node_embedding, children_embedding, children_index, node_dim, w_t, w_r, w_l, b_conv):
        """Convolve a batch of nodes and children.
        Lots of high dimensional tensors in this function. Intuitively it makes
        more sense if we did this work with while loops, but computationally this
        is more efficient. Don't try to wrap your head around all the tensor dot
        products, just follow the trail of dimensions.
        """
        with tf.compat.v1.name_scope('conv_step'):
            # nodes is shape (batch_size x max_tree_size x node_dim)
            # children is shape (batch_size x max_tree_size x max_children)

            with tf.compat.v1.name_scope('trees'):
              
                # add a 4th dimension to the parent nodes tensor
                # nodes is shape (batch_size x max_tree_size x 1 x node_dim)
                parent_node_embedding = tf.expand_dims(parent_node_embedding, axis=2)
                # tree_tensor is shape
                # (batch_size x max_tree_size x max_children + 1 x node_dim)
                tree_tensor = tf.concat([parent_node_embedding, children_embedding], axis=2, name='trees')

            with tf.compat.v1.name_scope('coefficients'):
                # coefficient tensors are shape (batch_size x max_tree_size x max_children + 1)
                c_t = self.eta_t(children_index)
                c_r = self.eta_r(children_index, c_t)
                c_l = self.eta_l(children_index, c_t, c_r)

                # concatenate the position coefficients into a tensor
                # (batch_size x max_tree_size x max_children + 1 x 3)
                coef = tf.stack([c_t, c_r, c_l], axis=3, name='coef')

            with tf.compat.v1.name_scope('weights'):
                # stack weight matrices on top to make a weight tensor
                # (3, node_dim, conv_output_dim)
                weights = tf.stack([w_t, w_r, w_l], axis=0)

            with tf.compat.v1.name_scope('combine'):
                batch_size = tf.shape(input=children_index)[0]
                max_tree_size = tf.shape(input=children_index)[1]
                max_children = tf.shape(input=children_index)[2]

                # reshape for matrix multiplication
                x = batch_size * max_tree_size
                y = max_children + 1
                result = tf.reshape(tree_tensor, (x, y, node_dim))
                coef = tf.reshape(coef, (x, y, 3))
                result = tf.matmul(result, coef, transpose_a=True)
                result = tf.reshape(result, (batch_size, max_tree_size, 3, node_dim))

                # output is (batch_size, max_tree_size, conv_output_dim)
                result = tf.tensordot(result, weights, [[2, 3], [0, 1]])

                # output is (batch_size, max_tree_size, conv_output_dim)

                # output = tf.nn.tanh(result + b_conv, name='conv')
                output = tf.nn.leaky_relu(result + b_conv)
                # output = tf.compat.v1.nn.swish(result + b_conv)
                # output = tf.layers.batch_normalization(output, training=self.placeholders['is_training'])
                output_drop_out = tf.nn.dropout(output, rate=self.placeholders["dropout_rate"])  # DROP-OUT here
                return output_drop_out

    def compute_children_node_types_tensor(self, parent_node_embedding, children_index, node_type_dim):
        """Build the children tensor from the input nodes and child lookup."""
    
        max_children = tf.shape(input=children_index)[2]
        batch_size = tf.shape(input=parent_node_embedding)[0]
        num_nodes = tf.shape(input=parent_node_embedding)[1]

        # replace the root node with the zero vector so lookups for the 0th
        # vector return 0 instead of the root vector
        # zero_vecs is (batch_size, num_nodes, 1)
        zero_vecs = tf.zeros((batch_size, 1, node_type_dim))
        # vector_lookup is (batch_size x num_nodes x node_dim)
        vector_lookup = tf.concat([zero_vecs, parent_node_embedding[:, 1:, :]], axis=1)
        # children is (batch_size x num_nodes x num_children x 1)
        children_index = tf.expand_dims(children_index, axis=3)
        # prepend the batch indices to the 4th dimension of children
        # batch_indices is (batch_size x 1 x 1 x 1)
        batch_index = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1, 1))
        # batch_indices is (batch_size x num_nodes x num_children x 1)
        batch_index = tf.tile(batch_index, [1, num_nodes, max_children, 1])
        # children is (batch_size x num_nodes x num_children x 2)
        children_index = tf.concat([batch_index, children_index], axis=3)
        # output will have shape (batch_size x num_nodes x num_children x node_type_dim)
        # NOTE: tf < 1.1 contains a bug that makes backprop not work for this!
        return tf.gather_nd(vector_lookup, children_index)


    def compute_parent_node_types_tensor(self, parent_node_type_index, node_type_embedding):
        parent_node_types_tensor =  tf.nn.embedding_lookup(params=node_type_embedding,ids=parent_node_type_index)
        return parent_node_types_tensor
    
    def compute_parent_node_tokens_tensor(self, parent_node_tokens_index, node_token_embedding):
        parent_node_tokens_tensor = tf.nn.embedding_lookup(params=node_token_embedding, ids=parent_node_tokens_index)
        parent_node_tokens_tensor = tf.reduce_sum(input_tensor=parent_node_tokens_tensor, axis=2)
        return parent_node_tokens_tensor

    # def compute_children_node_types_tensor(self, children_node_types_indices):
    #     children_node_types_tensor =  tf.nn.embedding_lookup(self.node_type_embeddings, children_node_types_indices)
    #     return children_node_types_tensor
    
    def compute_children_node_tokens_tensor(self, children_node_token_index, node_token_dim, node_token_embedding):
        batch_size = tf.shape(input=children_node_token_index)[0]
        zero_vecs = tf.zeros((1, node_token_dim))
        vector_lookup = tf.concat([zero_vecs, node_token_embedding[1:, :]], axis=0)
        children_node_tokens_tensor = tf.nn.embedding_lookup(params=vector_lookup, ids=children_node_token_index)
        children_node_tokens_tensor = tf.reduce_sum(input_tensor=children_node_tokens_tensor, axis=3)
        return children_node_tokens_tensor

    def eta_t(self, children):
        """Compute weight matrix for how much each vector belongs to the 'top'"""
        with tf.compat.v1.name_scope('coef_t'):
            # children is shape (batch_size x max_tree_size x max_children)
            batch_size = tf.shape(input=children)[0]
            max_tree_size = tf.shape(input=children)[1]
            max_children = tf.shape(input=children)[2]
            # eta_t is shape (batch_size x max_tree_size x max_children + 1)
            return tf.tile(tf.expand_dims(tf.concat(
                [tf.ones((max_tree_size, 1)), tf.zeros((max_tree_size, max_children))],
                axis=1), axis=0,
            ), [batch_size, 1, 1], name='coef_t')

    def eta_r(self, children, t_coef):
        """Compute weight matrix for how much each vector belogs to the 'right'"""
        with tf.compat.v1.name_scope('coef_r'):
            # children is shape (batch_size x max_tree_size x max_children)
            children = tf.cast(children, tf.float32)
            batch_size = tf.shape(input=children)[0]
            max_tree_size = tf.shape(input=children)[1]
            max_children = tf.shape(input=children)[2]

            # num_siblings is shape (batch_size x max_tree_size x 1)
            num_siblings = tf.cast(
                tf.math.count_nonzero(children, axis=2, keepdims=True),
                dtype=tf.float32
            )
            # num_siblings is shape (batch_size x max_tree_size x max_children + 1)
            num_siblings = tf.tile(
                num_siblings, [1, 1, max_children + 1], name='num_siblings'
            )
            # creates a mask of 1's and 0's where 1 means there is a child there
            # has shape (batch_size x max_tree_size x max_children + 1)
            mask = tf.concat(
                [tf.zeros((batch_size, max_tree_size, 1)),
                 tf.minimum(children, tf.ones(tf.shape(input=children)))],
                axis=2, name='mask'
            )

            # child indices for every tree (batch_size x max_tree_size x max_children + 1)
            child_indices = tf.multiply(tf.tile(
                tf.expand_dims(
                    tf.expand_dims(
                        tf.range(-1.0, tf.cast(max_children, tf.float32), 1.0, dtype=tf.float32),
                        axis=0
                    ),
                    axis=0
                ),
                [batch_size, max_tree_size, 1]
            ), mask, name='child_indices')

            # weights for every tree node in the case that num_siblings = 0
            # shape is (batch_size x max_tree_size x max_children + 1)
            singles = tf.concat(
                [tf.zeros((batch_size, max_tree_size, 1)),
                 tf.fill((batch_size, max_tree_size, 1), 0.5),
                 tf.zeros((batch_size, max_tree_size, max_children - 1))],
                axis=2, name='singles')

            # eta_r is shape (batch_size x max_tree_size x max_children + 1)
            return tf.compat.v1.where(
                tf.equal(num_siblings, 1.0),
                # avoid division by 0 when num_siblings == 1
                singles,
                # the normal case where num_siblings != 1
                tf.multiply((1.0 - t_coef), tf.divide(child_indices, num_siblings - 1.0)),
                name='coef_r'
            )

    def eta_l(self, children, coef_t, coef_r):
        """Compute weight matrix for how much each vector belongs to the 'left'"""
        with tf.compat.v1.name_scope('coef_l'):
            children = tf.cast(children, tf.float32)
            batch_size = tf.shape(input=children)[0]
            max_tree_size = tf.shape(input=children)[1]
            # creates a mask of 1's and 0's where 1 means there is a child there
            # has shape (batch_size x max_tree_size x max_children + 1)
            mask = tf.concat(
                [tf.zeros((batch_size, max_tree_size, 1)),
                    tf.minimum(children, tf.ones(tf.shape(input=children)))],
                axis=2,
                name='mask'
            )

            # eta_l is shape (batch_size x max_tree_size x max_children + 1)
            return tf.multiply(
                tf.multiply((1.0 - coef_t), (1.0 - coef_r)), mask, name='coef_l'
            )

   