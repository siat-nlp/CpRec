import tensorflow as tf
import ops_compress
from BlockWiseEmbedding import BlockWiseEmbeddingForInput as bl
from BlockWiseEmbedding import BlockWiseEmbeddingForSoftmax as bs
import lowranksoftmax
import time

class NextItNet_Decoder:

    def __init__(self, model_para):
        self.model_para = model_para
        embedding_width =  model_para['in_embed_size']
        output_dim =  model_para['dilated_channels']
        # embedding
        # self.allitem_embeddings = tf.get_variable('allitem_embeddings',
        #                                             [model_para['item_size'], embedding_width],
        #                                             initializer=tf.truncated_normal_initializer(stddev=0.02))

        # block-wise embedding, factor :1 means embeddings and other means block-wise embedding

        if (model_para['SoftmaxType'] == 'Block_Input_Full' or model_para['SoftmaxType'] == 'Block_Input_Softmax'
            or model_para['SoftmaxType']=='Block_Input_Softmax_Inference') and model_para['factor'] != 1:
            self.allitem_embeddings = bl(model_para['item_size'], embedding_width,
                                         block_factor=model_para['factor'],
                                         block=model_para["block"])
            print("using block embedding for input")
        else:
            self.allitem_embeddings = bl(model_para['item_size'], embedding_width,
                                         block_factor=1,
                                         block=model_para["block"])
            print("using embedding")

        self.allitem_embeddings.build()


    def train_graph(self):  #, is_negsample=False):
        model_para = self.model_para
        self.itemseq_input = tf.placeholder('int32',
                                         [model_para['batch_size'], model_para['seq_len']], name='itemseq_input')
        label_seq, dilate_input=self.model_graph(self.itemseq_input, train=True)

        # dilate_input : [batch_size, seq_len, dilated_channels]

        if model_para['SoftmaxType'] == "neg":
            print("using neg")
            logits_2D = tf.reshape(dilate_input, [-1,model_para['dilated_channels']])
            self.softmax_w = tf.get_variable("softmax_w", [model_para['item_size'],  model_para['dilated_channels']],tf.float32,tf.random_normal_initializer(0.0, 0.01))
            self.softmax_b = tf.get_variable("softmax_b", [model_para['item_size']], tf.float32, tf.constant_initializer(0.1))
            label_flat = tf.reshape(label_seq, [-1, 1])  # 1 is the number of positive example
            num_sampled = int(0.2* model_para['item_size'])#sample 20% as negatives
            # tf.nn.nce_loss
            loss =tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, label_flat, logits_2D, num_sampled, model_para['item_size'])
        elif model_para['SoftmaxType'] == "FullSoftmax_conv":
            print("using FullSoftmax_conv")
            if model_para['dilated_channels']!= model_para['out_embed_size']:
                self.softmax_pro_w = tf.get_variable("softmax_pro_w", [model_para['dilated_channels'], model_para['embed_size']],
                                                 tf.float32, tf.random_normal_initializer(0.0, 0.01))
                dilate_input = tf.tensordot(dilate_input, self.softmax_pro_w, axes=1)

            logits = ops_compress.conv1d(tf.nn.relu(dilate_input), model_para['item_size'], name='logits')
            logits_2D = tf.reshape(logits, [-1, model_para['item_size']])
            label_flat = tf.reshape(label_seq, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2D)
        elif model_para['SoftmaxType'] == "FullSoftmax" or model_para['SoftmaxType'] == "Block_Input_Full":
            print("using FullSoftmax")
            if model_para['dilated_channels']!= model_para['out_embed_size']:
                self.softmax_pro_w = tf.get_variable("softmax_pro_w", [model_para['dilated_channels'], model_para['out_embed_size']],
                                                 tf.float32, tf.random_normal_initializer(0.0, 0.01))
                dilate_input = tf.tensordot(dilate_input, self.softmax_pro_w, axes=1)

            self.softmax_w = tf.get_variable("softmax_w", [model_para['out_embed_size'], model_para['item_size']],
                                             tf.float32, tf.random_normal_initializer(0.0, 0.01))
            self.softmax_b = tf.get_variable("softmax_b", [model_para['item_size']], tf.float32,
                                             tf.constant_initializer(0.1))

            label_flat = tf.reshape(label_seq, [-1])
            logits_2D = tf.reshape(dilate_input, [-1, model_para['out_embed_size']])
            logits_2D = tf.matmul(logits_2D, self.softmax_w)
            logits_2D = tf.nn.bias_add(logits_2D, self.softmax_b)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2D)
        elif model_para['SoftmaxType'] == "FullSoftmax_Tied":
            print("using FullSoftmax_Tied")
            if model_para['dilated_channels']!= model_para['embed_size']:
                self.softmax_pro_w = tf.get_variable("softmax_pro_w", [model_para['dilated_channels'], model_para['embed_size']],
                                                 tf.float32, tf.random_normal_initializer(0.0, 0.01))
                dilate_input = tf.tensordot(dilate_input, self.softmax_pro_w, axes=1)

            self.softmax_w = tf.transpose(self.allitem_embeddings.embedding)
            # self.softmax_b = tf.get_variable("softmax_b", [model_para['item_size']], tf.float32,
            #                                  tf.constant_initializer(0.1))

            label_flat = tf.reshape(label_seq, [-1])
            logits_2D = tf.reshape(dilate_input, [-1, model_para['dilated_channels']])
            logits_2D = tf.matmul(logits_2D, self.softmax_w)
            # logits_2D = tf.nn.bias_add(logits_2D, self.softmax_b)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2D)
        elif model_para['SoftmaxType'] == "Block_for_Softmax":
            print("using Block_for_Softmax")
            logits_2D =  tf.reshape(dilate_input, [-1, model_para['dilated_channels']])
            block = model_para['block']
            assert model_para['dilated_channels'] == model_para['out_embed_size']
            softmax_layer = bs(input_dim=model_para['dilated_channels'], block= block,
                               block_factor=model_para['factor'])
            loss, _ = softmax_layer.loss(logits_2D, tf.reshape(label_seq, [-1]), "loss")
        elif (model_para['SoftmaxType'] == 'Block_Input_Softmax'
              or model_para['SoftmaxType'] == 'Block_Input_Softmax_Inference') and model_para['factor'] != 1:
            print("using Block_Input_Softmax")
            logits_2D =  tf.reshape(dilate_input, [-1, model_para['dilated_channels']])
            block = model_para['block']
            assert model_para['dilated_channels'] == model_para['out_embed_size']
            softmax_layer = bs(input_dim=model_para['dilated_channels'], block= block,
                               block_factor=model_para['factor'])
            loss, _ = softmax_layer.loss(logits_2D, tf.reshape(label_seq, [-1]), "loss")
        elif model_para['SoftmaxType'] == 'LowrankSoftmax' and model_para['factor'] != 1:
            print("using LowrankSoftmax")
            logits_2D = tf.reshape(dilate_input, [-1, model_para['dilated_channels']])
            block = model_para['block']
            softmax_layer = lowranksoftmax.LowRankSoftmax(input_dim=model_para['dilated_channels'],
                                                          block=block, block_factor=model_para['factor'])
            loss = softmax_layer.loss(logits_2D, tf.reshape(label_seq, [-1]), "loss")

        elif model_para['SoftmaxType'] == 'Block_Input_LowrankSoftmax' and model_para['factor'] != 1:
            print("using Block_Input_LowrankSoftmax")
            logits_2D = tf.reshape(dilate_input, [-1, model_para['dilated_channels']])
            block = model_para['block']
            softmax_layer = lowranksoftmax.LowRankSoftmax(input_dim=model_para['dilated_channels'],
                                                          block=block, block_factor=model_para['factor'])
            loss = softmax_layer.loss(logits_2D, tf.reshape(label_seq, [-1]), "loss")

        self.loss = tf.reduce_mean(loss)
        regularization = 0.001 * tf.reduce_mean([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.loss = self.loss + regularization

        # self.arg_max_prediction = tf.argmax(logits_2D, 1) #useless, if using negative sampling (i.e., negsample=True), it should be changed such as in predict_graph module

    def model_graph(self, itemseq_input, train=True):
        model_para = self.model_para
        context_seq = itemseq_input[:, 0:-1]
        label_seq = itemseq_input[:, 1:]

        print("context_seq: ", context_seq.shape)
        context_embedding = self.allitem_embeddings.get_input(context_seq)

        # self_attention
        # mask = self_attention.make_std_mask(context_seq, pad=model_para['pad'])
        # context_embedding = self_attention.attention(context_embedding, context_embedding, context_embedding, mask,
        #                                              dropout=0.5, train=train)
        dilate_input = context_embedding
        if (model_para['SoftmaxType'] == "FullSoftmax" or model_para['SoftmaxType'] == "FullSoftmax_conv" or
            model_para['SoftmaxType'] == "Block_for_Softmax") and \
                model_para['in_embed_size'] != model_para['dilated_channels']:
            embed_proj_w = tf.get_variable("embed_w", [model_para['in_embed_size'], model_para['dilated_channels']])
            dilate_input = tf.tensordot(dilate_input, embed_proj_w, axes=1)

        if model_para['parametersharing_type'] == 'original':
            for layer_id, dilation in enumerate(model_para['dilations']):
                dilate_input = ops_compress.nextitnet_residual_block(dilate_input, dilation,
                                                            layer_id, model_para['dilated_channels'],
                                                            model_para['kernel_size'], causal=True, train=train)
        elif model_para['parametersharing_type'] == 'cross-layer':
            for layer_id, dilation in enumerate(model_para['dilations']):
                dilate_input = ops_compress.nextitnet_residual_block_cross_layer(dilate_input, dilation,
                                                            layer_id, model_para['dilated_channels'],
                                                            model_para['kernel_size'], causal=True, train=train)
        elif model_para['parametersharing_type'] == 'cross-block':
            for layer_id, dilation in enumerate(model_para['dilations']):
                dilate_input = ops_compress.nextitnet_residual_block_cross_block(dilate_input, dilation,
                                                            layer_id, model_para['dilated_channels'],
                                                            model_para['kernel_size'], causal=True, train=train)
        elif model_para['parametersharing_type'] == 'adjacent-layer':
            for layer_id, dilation in enumerate(model_para['dilations']):
                dilate_input = ops_compress.nextitnet_residual_block_adjacent_layer(dilate_input, dilation,
                                                            layer_id, model_para['dilated_channels'],
                                                            model_para['kernel_size'], causal=True, train=train)
        elif model_para['parametersharing_type'] == 'adjacent-block':
            for layer_id, dilation in enumerate(model_para['dilations']):
                dilate_input = ops_compress.nextitnet_residual_adjacent_block(dilate_input, dilation,
                                                            layer_id, model_para['dilated_channels'],
                                                            model_para['kernel_size'], causal=True, train=train)
        return label_seq, dilate_input



    # output top-n based on recalled items instead of all items. You can use this interface for practical recommender systems.
    def predict_graph_onrecall(self, reuse=False): #is_negsample=False,
        if reuse:
            tf.get_variable_scope().reuse_variables()
        model_para = self.model_para #
        self.input_predict = tf.placeholder('int32', [model_para['batch_size'], model_para['seq_len']], name='input_predict')
        self.input_recall = tf.placeholder('int32', [model_para['batch_size'], model_para['seq_len']], name='input_recall')# candidate items

        label_seq, dilate_input = self.model_graph(self.input_predict, train=False)
        # label_flat = tf.reshape(label_seq[:, -1:], [-1]) # [batch_size]

        if model_para['SoftmaxType'] == 'neg_nowork':
            logits_2D=dilate_input[:, -1:, :]
            recall_mat = tf.nn.embedding_lookup(self.softmax_w, self.input_recall)
            logits_2D = tf.matmul(logits_2D, tf.transpose(recall_mat,[0,2,1]))
            logits_2D=tf.reshape(logits_2D, [-1, tf.shape(self.input_recall)[1]])
            recall_bias = tf.nn.embedding_lookup(self.softmax_b, self.input_recall)
            logits_2D=tf.add(logits_2D,recall_bias)
            probs_flat = tf.nn.softmax(logits_2D, name='softmax')
        elif model_para['SoftmaxType'] == 'neg':
            logits_2D = tf.reshape(dilate_input[:, -1:, :], [-1, model_para['out_embed_size']])
            logits_2D = tf.matmul(logits_2D, tf.transpose(self.softmax_w))
            logits_2D = tf.nn.bias_add(logits_2D, self.softmax_b)
            probs_flat = tf.nn.softmax(logits_2D)
        elif model_para['SoftmaxType'] == 'FullSoftmax_conv':
            print("recall one valid using FullSoftmax_conv")
            if model_para['dilated_channels']!= model_para['embed_size']:
                dilate_input = tf.tensordot(dilate_input, self.softmax_pro_w, axes=1)

            logits = ops_compress.conv1d(tf.nn.relu(dilate_input[:, -1:, :]), model_para['item_size'], name='logits')
            logits_2D = tf.reshape(logits, [-1, model_para['item_size']]) #[batch_size, item_size]
            probs_flat = tf.nn.softmax(logits_2D, name='softmax')
        elif model_para["SoftmaxType"] == "FullSoftmax" or model_para['SoftmaxType'] == "Block_Input_Full" or model_para['SoftmaxType'] == "FullSoftmax_Tied":
            print("valid using FullSoftmax")
            if model_para['dilated_channels'] != model_para['out_embed_size']:
                dilate_input = tf.tensordot(dilate_input, self.softmax_pro_w, axes=1)

            logits_2D = tf.reshape(dilate_input[:, -1:, :], [-1, model_para['out_embed_size']])
            logits_2D = tf.matmul(logits_2D, self.softmax_w)
            logits_2D = tf.nn.bias_add(logits_2D, self.softmax_b)
            probs_flat = tf.nn.softmax(logits_2D)
        elif model_para["SoftmaxType"] == "Block_for_Softmax":
            print("recall one valid using Block_for_Softmax")
            logits_2D = tf.reshape(dilate_input[:, -1:, :], [-1, model_para['dilated_channels']]) #[batch_size, dilated_channels]
            block = model_para['block']
            softmax_layer = bs(model_para["dilated_channels"], block,
                               block_factor=model_para['factor'])
            # loss, _ = softmax_layer.loss(logits_2D, label_flat, train=False, name="loss")
            probs_flat = softmax_layer.softmax(logits_2D, name='softmax')
        elif model_para["SoftmaxType"] == "Block_Input_Softmax" and model_para['factor'] != 1:
            print("recall one valid using Block_Input_Softmax")
            logits_2D = tf.reshape(dilate_input[:, -1:, :], [-1, model_para['dilated_channels']])
            block = model_para['block']
            softmax_layer = bs(model_para["dilated_channels"], block,
                               block_factor=model_para['factor'])
            # loss, _ = softmax_layer.loss(logits_2D, label_flat, train=False, name="loss")
            probs_flat = softmax_layer.softmax(logits_2D, name='softmax')
        elif model_para["SoftmaxType"] == "LowrankSoftmax":
            print("recall one valid using LowrankSoftmax")
            logits_2D = tf.reshape(dilate_input[:, -1:, :], [-1, model_para['dilated_channels']])
            block = model_para['block']
            softmax_layer = lowranksoftmax.LowRankSoftmax(input_dim=model_para['dilated_channels'], block= block,
                                                          block_factor=model_para['factor'])
            # loss, _ = softmax_layer.loss(logits_2D, label_flat, train=False, name="loss")
            probs_flat = softmax_layer.softmax(logits_2D, name='softmax') # [batch_size, item_size]
        elif model_para["SoftmaxType"] == "Block_Input_LowrankSoftmax":
            print("recall one valid using Block_Input_LowrankSoftmax")
            logits_2D = tf.reshape(dilate_input[:, -1:, :], [-1, model_para['dilated_channels']])
            block = model_para['block']
            softmax_layer = lowranksoftmax.LowRankSoftmax(input_dim=model_para['dilated_channels'], block= block,
                                                          block_factor=model_para['factor'])
            # loss, _ = softmax_layer.loss(logits_2D, label_flat, train=False, name="loss")
            probs_flat = softmax_layer.softmax(logits_2D, name='softmax') # [batch_size, item_size]
        elif model_para["SoftmaxType"] == "Block_Input_Softmax_Inference" and model_para['factor'] != 1:
            print("recall one valid using Block_Input_Softmax")
            logits_2D = tf.reshape(dilate_input[:, -1:, :], [-1, model_para['dilated_channels']])
            block = model_para['block']
            softmax_layer = bs(model_para["dilated_channels"], block=block, block_factor=model_para['factor'])
            # loss, _ = softmax_layer.loss(logits_2D, label_flat, train=False, name="loss")
            probs_flat = softmax_layer.softmax_inference_top(logits_2D, name='softmax', top_v=model_para['top_k'])

        # self.loss_test = tf.reduce_mean(loss)

        self.g_probs = probs_flat
        # newly added for weishi, since each input is one user (i.e., a batch), in fact we just need to rank the first batch, the below code is to select top-5
        # self.top_k= tf.nn.top_k(self.g_probs[:,-1], k=model_para['top_k'],name='top-k')

        #be carefule with the top-k values since the index represents the orders of your recalled items but not the original order.
        # self.top_k = tf.nn.top_k(self.g_probs, k=model_para['top_k'], name='top-k') # [batch_size, top_k]
        # self.top_k=tf.gather(self.input_recall, tf.contrib.framework.argsort(self.g_probs),name='top-k')


    # output top-n based on recalled items instead of all items. You can use this interface for practical recommender systems.
    def predict_graph_onrecall_ori(self, is_negsample=False, reuse=False):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        model_para = self.model_para
        self.input_predict = tf.placeholder('int32', [model_para['batch_size'], model_para['seq_len']], name='input_predict')
        self.input_recall = tf.placeholder('int32', [model_para['batch_size'], model_para['seq_len']], name='input_recall')  # candidate items

        label_seq, dilate_input = self.model_graph(self.input_predict, train=False)


        if is_negsample:
            logits_2D = dilate_input[:, -1:, :]
            recall_mat = tf.nn.embedding_lookup(self.softmax_w, self.input_recall)
            logits_2D = tf.matmul(logits_2D, tf.transpose(recall_mat, [0, 2, 1]))
            logits_2D = tf.reshape(logits_2D, [-1, tf.shape(self.input_recall)[1]])
            recall_bias = tf.nn.embedding_lookup(self.softmax_b, self.input_recall)
            logits_2D = tf.add(logits_2D, recall_bias)

        else:
            # logits = ops.conv1d(tf.nn.relu(dilate_input), model_para['item_size'], name='logits')
            logits = ops_compress.conv1d(tf.nn.relu(dilate_input[:, -1:, :]), model_para['item_size'], name='logits')
            logits_2D = tf.reshape(logits, [-1, model_para['item_size']])

        probs_flat = tf.nn.softmax(logits_2D, name='softmax')

        self.g_probs = probs_flat
        # newly added for weishi, since each input is one user (i.e., a batch), in fact we just need to rank the first batch, the below code is to select top-5
        # self.top_k= tf.nn.top_k(self.g_probs[:,-1], k=model_para['top_k'],name='top-k')

        # be carefule with the top-k values since the index represents the orders of your recalled items but not the original order.
        self.top_k = tf.nn.top_k(self.g_probs, k=model_para['top_k'], name='top-k')
        # self.top_k=tf.gather(self.input_recall, tf.contrib.framework.argsort(self.g_probs),name='top-k')





















