# Trans-SVNet: Accurate Phase Recognition from Surgical Videos via Hybrid Embedding Aggregation Transformer

You can refer to https://github.com/YuemingJin/TMRNet for data pre-processing.

1. run train_embedding.py to train ResNet50
2. run generate_LFB.py to generate spatial embeddings
3. run tecno.py to train TCN
4. run trans_SV.py to train Transformer

Note: although TCN is trained using the whole video, no future information is considered for each mini-batch. Please refer to the TeCNO paper for details.

https://arxiv.org/abs/2003.10751

