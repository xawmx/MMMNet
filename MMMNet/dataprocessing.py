import pandas as pd
import csv
import numpy as np
import torch
from pytorch_transformers import BertModel, BertConfig, BertTokenizer
from torch import nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

"""
VGGNet
"""
vgg_model_1000 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
pre_file = torch.load('vgg19-dcbb9e9d.pth')
vgg_model_1000.load_state_dict(pre_file)


"""
BERT
"""
#download from hugging face
config_path = 'bert-base-chinese/config.json'
model_path = 'bert-base-chinese/pytorch_model.bin'
vocab_path = 'bert-base-chinese/vocab.txt'


class BertTextNet(nn.Module):
    def __init__(self, code_length):
        super(BertTextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained(config_path)
        self.textExtractor = BertModel.from_pretrained(
            model_path, config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size

        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        # output[0](batch size, sequence length, model hidden dimension)

        # features = self.fc(text_embeddings)
        # features = self.tanh(features)
        features = text_embeddings
        return features


textNet = BertTextNet(code_length=768)

tokenizer = BertTokenizer.from_pretrained(vocab_path)


## Read csv data set file
csv_1 = pd.read_csv('')

# csv_1.iloc[:2, :]
# csv_1 = csv_1.loc[0:4]
# csv_00 = csv_1.iloc[:, 0]
# csv_11 = csv_1.iloc[:, 1]
# train_muti = np.array(csv_1)
# train_muti_list = train_muti.tolist()
# train_muti_textlist = np.array(train_muti_list)[:, 0]
# train_muti_picturelist = np.array(train_muti_list)[:, 1]
# Array_Row_Number = len(train_muti_textlist)
# train_textlist_H = [train_muti_textlist[row-1] for row in range(Array_Row_Number)]
# print(train_muti_list)
# print(train_muti_textlist)
# print(train_muti_picturelist)
# print(Array_Row_Number)
for i in range(15):
    csv_batch = csv_1.loc[10*i+15*39:10*(i+1)-1+15*39]
    # csv_batch_text = csv_batch.iloc[:, 0]
    # csv_batch_image = csv_batch.iloc[:, 1]
    train_batch = np.array(csv_batch)
    train_batch_list = train_batch.tolist()
    train_textlist = np.array(train_batch_list)[:, 0]
    train_imagelist = np.array(train_batch_list)[:, 1]
    Array_Row_Number = len(train_textlist)
    train_textlist_H = [train_textlist[row - 1] for row in range(Array_Row_Number)]
    # print(Array_Row_Number)
    # print(train_textlist_H)

    texts = np.array(train_textlist_H)
    tokens, segments, input_masks = [], [], []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)  #
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

    max_len = max([len(single) for single in tokens])

    for j in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        input_masks[j] += padding


    # PyTorch tensors
    tokens_tensor = torch.tensor(tokens)
    segments_tensors = torch.tensor(segments)
    input_masks_tensors = torch.tensor(input_masks)


    # get text features
    text_hashCodes = textNet(tokens_tensor, segments_tensors, input_masks_tensors)
    text_hashCodes = text_hashCodes.view(Array_Row_Number, 768)
    # text_hashCodes = text_hashCodes.view(1, 768)
    # print('文本特征：')
    # print(text_hashCodes)
    # print(text_hashCodes.shape)


    # Image data
    for rowpic in range(Array_Row_Number):
        pic_name = train_imagelist[rowpic - 1]
        image_path = 'path' + pic_name
        im = Image.open(image_path).convert('RGB')
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        im = trans(im)
        im.unsqueeze_(dim=0)

        # get image features
        vgg_model_1000 = vgg_model_1000.eval()
        image_feature_1000 = vgg_model_1000(im).data[0]
        image_feature_1000 = image_feature_1000.view(1, 1000)
        # print('dim of vgg_model_1000: ', image_feature_1000.shape)
        # print(image_feature_1000)
        if rowpic == 0:
            # out_a = torch.cat([out, image_feature_1000], 0)
            out_a = image_feature_1000
        else:
            out_b = image_feature_1000
            out_a = torch.cat([out_a, out_b], 0)

    # print('视觉特征：')
    # print(out_a)
    # print(out_a.shape)
    fusion_feature = torch.cat([out_a, text_hashCodes], 1)
    # print(fusion_feature)
    # print(fusion_feature.shape)
    if i == 0:
        feature_out = fusion_feature
    else:
        feature_out_1 = fusion_feature
        feature_out = torch.cat([feature_out, feature_out_1], 0)


print(feature_out)
print(feature_out.shape)
feature_save = feature_out.detach().numpy()
np.save('name.npy', )