You can email me if you have any questions about the performance and efficiency of CpRec.  

```
you can run nextitrec_eval.py (python nextitrec_eval.py) directly, which includes training and testing.  
Each training epoch, test once test set.  
```

## The import configuration:  
```
--softmax_type: This is used to control whether block-wise embedding is used. There are 5 classes:  
FullSoftmax：The input layer and output layer use basic embedding and softmax respectively.  
Block_Input_Full：Only the input layer uses block-wise embedding, and the output layer uses basic softmax.  
Block_for_Softmax：Only the output layer uses block-wise embedding, and the output layer uses basic softmax.  
Block_Input_Softmax：Both the input layer and the output layer use block-wise embedding.  
Block_Input_Softmax_Inference： Both the input layer and the output layer use block-wise embedding, but fast inference is used in the prediction stage. it should be noted that the use of this type requires batch_size_test = 1  
```
```
--use_parametersharing_type: This is used to control whether to use layer-wise parameter sharing, there are 5 classes:  
original：No parameter sharing.  
cross-layer：use the cross-layer parameter sharing.  
cross-block：use the cross-block parameter sharing.  
adjacent-layer：use the adjacent-layer parameter sharing.  
adjacent-block：use the adjacent-block parameter sharing.  
```
you should cross-valid the number of blocks and the embedding size of each block, but we strongly suggest you use 3~5 blocks.  
--use_embedding_type_factor is the attenuation factor of block dimension in block-wise embedding, the default is 4. You can also manually set the dimension of each block, e.g. [300, 200, 100].  

## Dataset
you can download a large sequential dataset of movielen that has been pre-processed: https://pan.baidu.com/s/1XZCUoRwWFa8fpRWpEA2HPQ  
code (提取码):fo51  

# References
https://github.com/fajieyuan/nextitnet code that including Caser and GRURec  
https://github.com/graytowne/caser code  

# CpRec
A Generic Network Compression Framework for Sequential Recommender Systems  

Please cite this paper if you find our code is useful.  
