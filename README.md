# Multimodal fake news detection and risk warning


# Dataset
Multimodal fake news detection: Weibo and Twitter

You can download Weibo dataset from https://pan.baidu.com/s/1Vn75mXe69jC9txqB81QzUQ ( extraction code: 78uo )
or https://drive.google.com/file/d/14VQ7EWPiFeGzxp3XC2DeEHi-BEisDINn/view?usp=sharing.

You can download Twitter dataset from https://github.com/MKLab-ITI/image-verification-corpus/tree/master/mediaeval2016
or https://github.com/plw-study/MRML

Multimodal news risk warning: SM-D, GenIamge, NELA

SM-D: https://www.selectdataset.com/dataset/beaf1420934ba365a1e37ed6f0412f6f

GenIamge: https://github.com/GenImage-Dataset/GenImage

NELA: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/AMCV2H

# Preprocessing
After setting the path of the dataset, you can use dataprocessing for data preprocessing.

# Train
After the first phase of training is completed, the .pth file will be saved, and then the second phase of classifier training needs to be manually performed.

# Test
The test codes for all indicators are in the train.py file. You can select the evaluation indicators for output.


## Requirements
- Python 3.9
- Pytorch 1.13.1
- Math
- BERT (hugging-face https://huggingface.curated.co/)
- VGG 19 ("https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs" or 'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
