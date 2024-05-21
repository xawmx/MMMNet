import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
from model import Alignment, Classifier
from mask_generator import FeatureMask
from tqdm import tqdm


# Configs
BATCH_SIZE = 32
AUXILIARY_EPOCH_NUM = 60
CLASSIFIER_EPOCH_NUM = 60


class FeatureDataset(Dataset):
    def __init__(self, text, pic, labels):
        self.text = text
        self.pic = pic
        # self.test_labels = torch.from_numpy(text).long()
        self.labels = labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return self.text[item], self.pic[item], self.labels[item]


def train():
    # ---  Load Config  ---
    batch_size = BATCH_SIZE
    auxiliary_epoch_num = AUXILIARY_EPOCH_NUM
    classifier_epoch_num = CLASSIFIER_EPOCH_NUM

    # ---  Load Data  ---
    # The feature vectors we processed are saved in npy files
    pic_sample = np.load('')
    text_fixed = np.load('')
    pic_matched = np.load('')
    text_fake = np.load('')
    pic_fake = np.load('')

    pic_sample_tensor = torch.tensor(pic_sample, dtype=torch.float32)
    text_fixed_tensor = torch.tensor(text_fixed, dtype=torch.float32)
    pic_matched_tensor = torch.tensor(pic_matched, dtype=torch.float32)
    text_fake_tensor = torch.tensor(text_fake, dtype=torch.float32)
    pic_fake_tensor = torch.tensor(pic_fake, dtype=torch.float32)

    labels_1 = torch.ones([m1, 1], dtype=torch.float32)
    labels_2 = torch.ones([m2, 1], dtype=torch.float32)
    auxiliary_task_labels = torch.cat([labels_1, -1 * labels_2], 0)
    text = torch.cat([text_fixed_tensor, text_fake_tensor], 0)
    pic = torch.cat([pic_matched_tensor, pic_fake_tensor], 0)
    labels_3 = torch.ones([n1, 1], dtype=torch.float32)
    labels_4 = torch.zeros([n2, 1], dtype=torch.float32)
    classifier_task_label = torch.cat([labels_3, labels_4], 0)

    # auxiliary_task_text = torch.cat([text_fixed_tensor, text_fixed_tensor], 0)
    # auxiliary_task_pic = torch.cat([pic_matched_tensor, pic_sample_tensor], 0)

    auxiliary_task_train_dataset = FeatureDataset(text_fixed_tensor, pic_sample_tensor, auxiliary_task_labels)
    classifier_task_train_dataset = FeatureDataset(text, pic, classifier_task_label)

    train_set, test_set = torch.utils.data.random_split(classifier_task_train_dataset, [10000, 2000])

    auxiliary_task_train_dataloader = DataLoader(auxiliary_task_train_dataset, batch_size=batch_size, shuffle=True)
    classifier_task_train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

    # ---  Build Model & Trainer & Loss Function & Optimizer  ---
    alignment_model = Alignment()
    # auxiliary_task_model = Auxiliary_Task()
    classifier_task_model = Classifier()
    auto_mask = FeatureMask(80)

    loss_function_auxiliary = nn.CosineEmbeddingLoss(reduction='mean')
    loss_function_classifier = nn.BCELoss(reduction='mean')

    optim_alignment = torch.optim.Adam(
        alignment_model.parameters(), lr=1e-4, weight_decay=0
    )
    # optim_auxiliary_task = torch.optim.Adam(
    #     auxiliary_task_model.parameters(), lr=1e-3, weight_decay=0
    # )
    optim_classifier_task = torch.optim.Adam(
        classifier_task_model.parameters(), lr=1e-5, weight_decay=0
    )
    optim_mask = torch.optim.Adam(
        auto_mask.parameters(), lr=1e-4, weight_decay=0
    )

    # ---  Models Training  ---
    # loss_auxiliary_total = 0
    # loss_classifier_total = 0
    # best_acc = 0
    for epoch_auxiliary in range(auxiliary_epoch_num):
        alignment_model.train()
        # auxiliary_task_model.train()
        classifier_task_model.train()
        auto_mask.train()
        corrects_pre_similarity = 0
        corrects_pre_classifier = 0
        loss_auxiliary_total = list()
        loss_classifier_total = 0
        similarity_count = 0
        classifier_count = 0

        # ---  AUXILIARY TASK  ---
        for text, pic, label in auxiliary_task_train_dataloader:
            text_aligned, pic_aligned = alignment_model(text, pic)
            label_reshaped = torch.reshape(label, (-1,))
            # _, similar = classifier_task_model(text_aligned, pic_aligned)
            loss_auxiliary = loss_function_auxiliary(text_aligned, pic_aligned, label_reshaped)
            optim_alignment.zero_grad()
            # optim_auxiliary_task.zero_grad()
            loss_auxiliary.backward()
            optim_alignment.step()
            # optim_auxiliary_task.step()
            loss_auxiliary_total.append(loss_auxiliary.item())

        print(f"Epoch {epoch_auxiliary + 1}/{auxiliary_epoch_num}, Loss: {np.mean(np.array(loss_auxiliary_total)):.4f}")
        alignment_model.eval()
        torch.save(alignment_model, 'align.pth')


        # auxiliary_task_model.eval()

        # # ---  CLASSIFIER TASK  ---
        # for text, pic, label in classifier_task_train_dataloader:
        #     text_align, pic_align = alignment_model(text, pic)
        #     aligned_feature, _ = auxiliary_task_model(text_align, pic_align)
        #     label_pre = classifier_task_model(text_align, pic_align, text, pic)
        #     loss_classifier = loss_function_classifier(label_pre, label)
        #     optim_classifier_task.zero_grad()
        #     loss_classifier.backward()
        #     optim_classifier_task.step()
        #     loss_classifier_total += loss_classifier
        # print(f"Epoch {epoch_auxiliary + 1}/{auxiliary_epoch_num}, Loss: {loss_classifier_total:.4f}")

    alignment_model = torch.load('align.pth')

    for epoch_classifier in range(classifier_epoch_num):
        alignment_model.eval()
        # auxiliary_task_model.eval()
        classifier_task_model.train()
        # auto_mask.train()
        # mask_1 = auto_mask()
        loss_classifier_total = 0
        # ---  CLASSIFIER TASK  ---
        for text, pic, label in classifier_task_train_dataloader:
            text_align, pic_align = alignment_model(text, pic)
            label_pre = classifier_task_model(text_align, pic_align, text, pic)
            loss_classifier = loss_function_classifier(label_pre, label)
            optim_classifier_task.zero_grad()
            optim_mask.zero_grad()
            loss_classifier.backward()
            optim_classifier_task.step()

            # # ---  update dimensional mask  ---
            # grad_1 = torch.autograd.grad(loss_classifier, mask_1, create_graph=True)[0]
            # grad_2 = torch.autograd.grad(grad_1, mask_1, grad_outputs=torch.ones_like(grad_1))[0]
            # auto_mask.grad = grad_2
            # optim_mask.step()
            loss_classifier_total += loss_classifier

        with torch.no_grad():
            correct = 0
            total = 0
            for t, p, l in test_dataloader:
                t_a, p_a = alignment_model(t, p)
                l_pre = classifier_task_model(t_a, p_a, t, p)
                predicted = l_pre > 0.5
                total += l.size(0)
                correct += (predicted == l).sum().item()
            acc = 100 * correct / total
            print(f"Test Accuracy: {acc:.2f}%")

        print(f"Epoch {epoch_classifier + 1}/{classifier_epoch_num}, Loss: {loss_classifier_total:.4f}")


    alignment_model.eval()
    classifier_task_model.eval()
    # torch.save(classifier_task_model, 'class.pth')
    auto_mask.eval()
    # # ---  Test  ---
    with torch.no_grad():
        correct = 0
        correct_p = 0
        correct_1 = 0
        total = 0
        for t, p, l in classifier_task_train_dataloader:
            t_a, p_a = alignment_model(t, p)
            l_pre = classifier_task_model(t_a, p_a, t, p, auto_mask)
            predicted = l_pre > 0.5
            total += l.size(0)
            correct += (predicted == l).sum().item()
            correct_p += (predicted == l & l == 1).sum().item()
            correct_1 += (predicted == 1).sum().item()
        acc = 100 * correct / total
        # pre = 100 * correct_p / n1
        # rec = 100 * correct_p / correct_1
        # f1_score = (pre * rec * 2) / (pre + rec)
        print(f"Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    train()