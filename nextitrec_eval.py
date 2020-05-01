import tensorflow as tf
import generator_recsys
import utils
import shutil
import time
import math
import numpy as np
import argparse
from Data_loader import Data_Loader
import os

# You can run it directly, first training and then evaluating
# nextitrec_generate.py can only be run when the model parameters are saved, i.e.,
#  save_path = saver.save(sess,
#                       "Data/Models/generation_model/model_nextitnet.ckpt".format(iter, numIters))
# if you are dealing very huge industry dataset, e.g.,several hundred million items, you may have memory problem during training, but it 
# be easily solved by simply changing the last layer, you do not need to calculate the cross entropy loss
# based on the whole item vector. Similarly, you can also change the last layer (use tf.nn.embedding_lookup or gather) in the prediction phrase 
# if you want to just rank the recalled items instead of all items. The current code should be okay if the item size < 5 million.



#Strongly suggest running codes on GPU with more than 10G memory!!!
#if your session data is very long e.g, >50, and you find it may not have very strong internal sequence properties, you can consider generate subsequences
def generatesubsequence(train_set):
    # create subsession only for training
    subseqtrain = []
    for i in range(len(train_set)):
        # print x_train[i]
        seq = train_set[i]
        lenseq = len(seq)
        # session lens=100 shortest subsession=5 realvalue+95 0
        for j in range(lenseq - 2):
            subseqend = seq[:len(seq) - j]
            subseqbeg = [0] * j
            subseq = np.append(subseqbeg, subseqend)
            # beginseq=padzero+subseq
            # newsubseq=pad+subseq
            subseqtrain.append(subseq)
    x_train = np.array(subseqtrain)  # list to ndarray
    del subseqtrain
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_train = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_train]
    print("generating subsessions is done!")
    return x_train


def INFO_LOG(info):
    print("[%s]%s"%(time.strftime("%Y-%m-%d %X", time.localtime()), info))
    # logging.info("[%s]%s"%(time.strftime("%Y-%m-%d %X", time.localtime()), info))

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--datapath', type=str, default='Data/Session/mllatest_ls20.csv',
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=5000,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=10000,
                        help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--is_generatesubsession', type=bool, default=False,
                        help='whether generating a subsessions, e.g., 12345-->01234,00123,00012  It may be useful for very some very long sequences')
    parser.add_argument('--use_softmax_type', type=str, default="Block_Input_Softmax",
                        help="using FullSoftmax/Block_Input_Full/Block_for_Softmax/Block_Input_Softmax/Block_Input_Softmax_Inference")
    # Block_Input_Softmax_Inference is the fast inference and the batch_size_test must be 1
    parser.add_argument('--use_embedding_type_factor', type=int, default=4,
                        help="using block-wise embedding shift factor, but 1 means basic embedding")
    parser.add_argument('--use_parametersharing_type', type=str, default="original",
                        help="using original/cross-layer/cross-block/adjacent-layer/adjacent-block")
    args = parser.parse_args()

	print(args)

    dl = Data_Loader({'model_type': 'generator', 'dir_name': args.datapath})
    all_samples = dl.items
    items_voc = dl.item2id


    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]

    print("shape: ", np.shape(all_samples))
    # logging.info("shape: {}".format(np.shape(all_samples)))

    # Split train/test set
    dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
    train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]

    if args.is_generatesubsession:
        x_train = generatesubsequence(train_set)

    model_para = {
        #if you changed the parameters here, also do not forget to change paramters in nextitrec_generate.py
        'item_size': len(items_voc),
        'in_embed_size': 512,
        'dilated_channels': 512,
        'out_embed_size': 512,
        # if you use nextitnet_residual_block, you can use [1, 4, ],
        # if you use nextitnet_residual_block_one, you can tune and i suggest [1, 2, 4, ], for a trial
        # when you change it do not forget to change it in nextitrec_generate.py
        'dilations': [1, 4, 1, 4],
        'kernel_size': 3,
        'learning_rate':0.001,
        'batch_size':128,
        'iterations':500,
        'is_negsample':False, #False denotes no negative sampling
        'SoftmaxType':args.use_softmax_type,
        'block': [7000, 15000, len(items_voc)],
        'factor': args.use_embedding_type_factor,
        'seq_len': len(all_samples[0]),
        'pad': dl.padid,
        'parametersharing_type': args.use_parametersharing_type,
    }
    print("in_embed_size", model_para["in_embed_size"])
    print("dilated_channels", model_para["dilated_channels"])
    print("out_embed_size", model_para["out_embed_size"])
    print("dilations", model_para['dilations'])
    print("batch_size", model_para["batch_size"])
    print("block", model_para["block"])
    print("factor", model_para["factor"])
    print("parametersharing_type", model_para["parametersharing_type"])

    # print("seq_len: ", model_para['seq_len'])
    itemrec = generator_recsys.NextItNet_Decoder(model_para)
    itemrec.train_graph() # model_para['is_negsample'])
    optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1).minimize(itemrec.loss)
    itemrec.predict_graph_onrecall(reuse=True) # model_para['is_negsample'],)

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    sess= tf.Session(config=session_config)
    init=tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()


    for iter in range(model_para['iterations']):
        batch_no = 0
        batch_size = model_para['batch_size']
        start = time.time()
        batch_num = train_set.shape[0] / batch_size
        INFO_LOG("-------------------------------------------------------train1")
        while (batch_no + 1) * batch_size < train_set.shape[0]:

            item_batch = train_set[batch_no * batch_size: (batch_no + 1) * batch_size, :]
            _, loss = sess.run(
                [optimizer, itemrec.loss],
                feed_dict={
                    itemrec.itemseq_input: item_batch
                })
            if batch_no % max(10, batch_num//10) == 0:
                INFO_LOG("{}/{} Train LOSS: {}\tepoch: {}\t total_epoch:{}\t total_batches:{}".format(
                    batch_no,  batch_num, loss, iter, model_para['iterations'], batch_num))
            batch_no += 1

        end = time.time()

        INFO_LOG("Train LOSS: {}\tepoch: {}\t total_epoch:{}\t total_batches:{}".format(
            loss, iter, model_para['iterations'], batch_num))
        INFO_LOG("TIME FOR EPOCH: {}".format(end - start))
        INFO_LOG("TIME FOR BATCH (mins): {}".format((end - start) / batch_num))

        INFO_LOG("-------------------------------------------------------test1")
        batch_no_test = 0
        batch_size_test = batch_size*1
        # if you need use fast inference, please use batch_size_test=1
        batch_num_test = valid_set.shape[0] / batch_size_test
        curr_preds_5=[]
        rec_preds_5=[] #1
        ndcg_preds_5=[] #1
        curr_preds_20 = []
        rec_preds_20 = []  # 1
        ndcg_preds_20  = []  # 1
        test_start = time.time()
        while (batch_no_test + 1) * batch_size_test < valid_set.shape[0]:
            item_batch = valid_set[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
            [probs] = sess.run( # , loss_test
                [itemrec.g_probs], # , itemrec.loss_test
                feed_dict={
                    itemrec.input_predict: item_batch
                })

            for bi in range(probs.shape[0]):

                pred_items_5 = utils.sample_top_k(probs[bi], top_k=args.top_k)  # top_k=5
                pred_items_20 = utils.sample_top_k(probs[bi], top_k=args.top_k + 15)

                true_item=item_batch[bi][-1]
                predictmap_5 = {ch: i for i, ch in enumerate(pred_items_5)}
                pred_items_20 = {ch: i for i, ch in enumerate(pred_items_20)}

                rank_5=predictmap_5.get(true_item)
                rank_20 = pred_items_20.get(true_item)
                if rank_5 ==None:
                    curr_preds_5.append(0.0)
                    rec_preds_5.append(0.0)#2
                    ndcg_preds_5.append(0.0)#2
                else:
                    MRR_5 = 1.0/(rank_5+1)
                    Rec_5=1.0#3
                    ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
                    curr_preds_5.append(MRR_5)
                    rec_preds_5.append(Rec_5)#4
                    ndcg_preds_5.append(ndcg_5)  # 4
                if rank_20 ==None:
                    curr_preds_20.append(0.0)
                    rec_preds_20.append(0.0)#2
                    ndcg_preds_20.append(0.0)#2
                else:
                    MRR_20 = 1.0/(rank_20+1)
                    Rec_20=1.0#3
                    ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)  # 3
                    curr_preds_20.append(MRR_20)
                    rec_preds_20.append(Rec_20)#4
                    ndcg_preds_20.append(ndcg_20)  # 4

            if batch_no_test % max(10, batch_num_test//10) == 0:
                INFO_LOG("{}/{}\tepoch: {}\t total_epoch:{}\t total_batches:{}".format(
                            batch_no_test,  batch_num_test, iter, model_para['iterations'], batch_num_test))
                INFO_LOG("Accuracy hit_5: {}".format(sum(rec_preds_5) / float(len(rec_preds_5))))  # 5
                INFO_LOG("Accuracy hit_20: {}".format(sum(rec_preds_20) / float(len(rec_preds_20))))  # 5


            batch_no_test += 1

        INFO_LOG("epoch: {}\t total_epoch:{}\t total_batches:{}".format(
            iter, model_para['iterations'], valid_set.shape[0] / batch_size))
        INFO_LOG("Accuracy mrr_5: {}".format(sum(curr_preds_5) / float(len(curr_preds_5)))) # 5
        INFO_LOG("Accuracy mrr_20: {}".format(sum(curr_preds_20) / float(len(curr_preds_20))))  # 5
        INFO_LOG("Accuracy hit_5: {}".format(sum(rec_preds_5) / float(len(rec_preds_5))))  # 5
        INFO_LOG("Accuracy hit_20: {}".format(sum(rec_preds_20) / float(len(rec_preds_20))))  # 5
        INFO_LOG("Accuracy ndcg_5: {}".format(sum(ndcg_preds_5) / float(len(ndcg_preds_5))))  # 5
        INFO_LOG("Accuracy ndcg_20: {}".format(sum(ndcg_preds_20) / float(len(ndcg_preds_20))))  #
        test_end = time.time()
        INFO_LOG("TIME FOR TEST EPOCH: {}".format(test_end - test_start))


if __name__ == '__main__':
    main()
