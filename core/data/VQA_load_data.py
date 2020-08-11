from __future__ import absolute_import, division, print_function

import argparse
import os
import re
import sys
from collections import Counter

import en_vectors_web_lg
import numpy as np
import torch

from core.data.data_utils import img_feat_path_load, img_feat_load, ques_load, tokenize, ans_stat
from core.data.data_utils import proc_img_feat, proc_ques
from core.data.ans_punct import prep_ans

import glob, json, torch, time
import torch.utils.data as Data

# Constants in the vocabulary
UNK_WORD = "<unk>"
PAD_WORD = "<_>"
PAD = 0


def get_top_answers(examples, occurs=0):
    """
    Extract all of correct answers in the dataset. Build a set of possible answers which
    appear more than pre-defined "occurs" times.
    --------------------
    Arguments:
        examples (list): the json data loaded from disk.
        occurs (int): a threshold that determine which answers are kept.
    Return:
        vocab_ans (list): a set of correct answers in the dataset.
    """
    counter = Counter()
    for ex in examples:
        for ans in ex["mc_ans"]:
            ans = str(ans).lower()
            ans_proc = prep_ans(ans)
            counter.update([ans_proc])

    frequent_answers = list(filter(lambda x: x[1] > occurs, counter.items()))
    total_ans = sum(item[1] for item in counter.items())
    total_freq_ans = sum(item[1] for item in frequent_answers)

    print("Number of unique answers:", len(counter))
    print("Total number of answers:", total_ans)
    print("Top %i answers account for %f%%" % (len(frequent_answers), total_freq_ans * 100.0 / total_ans))
    print("Sample frequent answers:")
    print("\n".join(map(str, frequent_answers[:20])))

    vocab_ans = []
    for item in frequent_answers:
        vocab_ans.append(item[0])

    return vocab_ans


def filter_answers(examples, ans2idx):
    """
    Remove the answers that don't appear in our answer set.
    --------------------
    Arguments:
        examples (list): the json data that contains all of answers in the dataset.
        ans2idx (dict): a set of considered answers.
    Return:
        examples (list): the processed json data which contains only answers in the answer set.
    """
    for ex in examples:
        ex["ans_score"] = [list(filter(lambda x: prep_ans(x[0]) in ans2idx, answers)) for answers in ex["ans_score"]]

    return examples


def refine_data_oe(dataset):
    for index in range(len(dataset)):
        dataset[index]['ques_id'] = dataset[index]['ques_id'][0]
        dataset[index]['ques'] = dataset[index]['ques'][0]
        if 'mc_ans' in dataset[index].keys():
            dataset[index]['mc_ans'] = dataset[index]['mc_ans'][0]
            dataset[index]['ans_score'] = dataset[index]['ans_score'][0]
            dataset[index]['ans_freq'] = dataset[index]['ans_freq'][0]
    return dataset


# Use once at the first to generate the answer dict and remove the used answers from the dataset
def build_ans_dict_OE():
    # build the answer dict from the train-val split
    trainval_set = json.load(open("/ExpData/gwy/vqa/v1/vqa_train_val_OE.json", "r"))
    top_answers = get_top_answers(trainval_set, 8)
    num_ans = len(top_answers)
    ans2idx = {}
    for idx, ans in enumerate(top_answers):
        ans2idx[ans] = idx
    idx2ans = top_answers

    trainval_set = filter_answers(trainval_set, ans2idx)
    trainval_set = refine_data_oe(trainval_set)
    with open("/ExpData/gwy/vqa/v1/preprocessed/vqa_train_val_OE.json", "w+") as file:
        json.dump(obj=trainval_set, fp=file)
    with open("/ExpData/gwy/vqa/v1/preprocessed/ans_dict.json", "w+") as file:
        json.dump(obj=(ans2idx, idx2ans), fp=file)

    train_set = json.load(open("/ExpData/gwy/vqa/v1/vqa_train_oe.json", "r"))
    train_set = filter_answers(train_set, ans2idx)
    train_set = refine_data_oe(train_set)
    with open("/ExpData/gwy/vqa/v1/preprocessed/vqa_train_oe.json", "w+") as file:
        json.dump(obj=train_set, fp=file)

    print("Process val dataset...")
    val_set = json.load(open("/ExpData/gwy/vqa/v1/vqa_val_oe.json", "r"))
    val_set = filter_answers(val_set, ans2idx)
    val_set = refine_data_oe(val_set)
    with open("/ExpData/gwy/vqa/v1/preprocessed/vqa_val_oe.json", "w+") as file:
        json.dump(obj=val_set, fp=file)

    print("process testdev split...")
    testdev_set = json.load(open("/ExpData/gwy/vqa/v1/vqa_test_oe.json", "r"))
    testdev_set = refine_data_oe(testdev_set)
    with open("/ExpData/gwy/vqa/v1/preprocessed/vqa_test_oe.json", "w+") as file:
        json.dump(testdev_set, fp=file)

    print("process test split...")
    test_set = json.load(open("/ExpData/gwy/vqa/v1/vqa_testdev_oe.json", "r"))
    test_set = refine_data_oe(test_set)
    with open("/ExpData/gwy/vqa/v1/preprocessed/vqa_testdev_oe.json", "w+") as file:
        json.dump(test_set, fp=file)


# build_ans_dict_OE()

data_map_vqa_oe = {
    "train": "/ExpData/gwy/vqa/v1/preprocessed/vqa_train_oe.json",
    "val": "/ExpData/gwy/vqa/v1/preprocessed/vqa_val_oe.json",
    "trainval": "/ExpData/gwy/vqa/v1/preprocessed/vqa_train_val_OE.json",
    "testdev": "/ExpData/gwy/vqa/v1/preprocessed/vqa_testdev_oe.json",
    "test": "/ExpData/gwy/vqa/v1/preprocessed/vqa_test_oe.json",
}


def proc_ans_oe(ques, ans_to_ix):
    answers = ques['ans_score']

    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)

    for ans in answers:
        ans_proc = prep_ans(ans[0])
        ans_score[ans_to_ix[ans_proc]] = ans[1]

    return ans_score


class DataSetOE(Data.Dataset):

    def __init__(self, __C):
        self.__C = __C

        # Loading all image paths
        # if self.__C.PRELOAD:
        self.img_feat_path_list = []
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            if split in ['train', 'val', 'test']:
                self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH[split] + '*.npz')

        # Loading question word list
        self.stat_ques_list = \
            json.load(open(data_map_vqa_oe['train'], 'r')) + \
            json.load(open(data_map_vqa_oe['val'], 'r')) + \
            json.load(open(data_map_vqa_oe['test'], 'r')) + \
            json.load(open(data_map_vqa_oe['testdev'], 'r'))

        # Loading question and answer list
        self.ques_list = []
        # self.ans_list = []

        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(data_map_vqa_oe[split], 'r'))
            # if __C.RUN_MODE in ['train']:
            #     self.ans_list += json.load(open(data_map_vqa_oe[split], 'r'))['annotations']

        # Define run data size
        # if __C.RUN_MODE in ['train']:
        #     self.data_size = self.ans_list.__len__()
        # else:
        self.data_size = self.ques_list.__len__()

        print('== Dataset size:', self.data_size)

        # {image id} -> {image feature absolutely path}
        self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)

        # {question id} -> {question}
        self.qid_to_ques = ques_load(self.ques_list, element_name='ques_id')

        # Tokenize
        self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, __C.USE_GLOVE, element_name='ques')
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        self.ans_to_ix, self.ix_to_ans = ans_stat('/ExpData/gwy/vqa/v1/preprocessed/ans_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')

    def __getitem__(self, idx):

        # For code safety
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)
        pad = np.zeros(1)

        # Process ['train'] and ['val', 'test'] respectively
        if self.__C.RUN_MODE in ['train']:
            # Load the run data from list
            ques = self.ques_list[idx]
            # ques = self.qid_to_ques[str(current_ques['ques_id'])]

            # Process image feature from (.npz) file
            # CHANGED
            try:
                img_feat = np.load(self.iid_to_img_feat_path[str(ques['img_id'])])
                img_feat_x = img_feat['x'].transpose((1, 0))
                bboxes = img_feat['bbox']
                img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)
            except:
                print('false')
                print(self.iid_to_img_feat_path[str(ques['img_id'])])

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN, element_name='ques')

            # Process answer
            ans_iter = proc_ans_oe(ques, self.ans_to_ix)

        else:
            # Load the run data from list
            ques = self.ques_list[idx]

            img_feat = np.load(self.iid_to_img_feat_path[str(ques['img_id'])])
            img_feat_x = img_feat['x'].transpose((1, 0))
            bboxes = img_feat['bbox']
            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN, element_name='ques')

        return torch.from_numpy(img_feat_iter), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter), \
               ques['img_id'], \
               pad, \
               ques['ques_id'], \
               pad, pad, pad, pad
            # bboxes, \

    def __len__(self):
        return self.data_size


# #################################### MC

data_map_vqa_mc = {
    "train": "/ExpData/gwy/vqa/v1/preprocessed/vqa_train_MC.json",
    "val": "/ExpData/gwy/vqa/v1/preprocessed/vqa_val_MC.json",
    "trainval": "/ExpData/gwy/vqa/v1/preprocessed/vqa_train_val_MC.json",
    "testdev": "/ExpData/gwy/vqa/v1/preprocessed/vqa_testdev_MC.json",
    "test": "/ExpData/gwy/vqa/v1/preprocessed/vqa_test_MC.json",
    "train_comp_path": "/ExpData/gwy/vqa/v2_mscoco_train2014_complementary_pairs.json",
    "val_comp_path": "/ExpData/gwy/vqa/v2_mscoco_val2014_complementary_pairs.json",
}


def filter_answers_mc(examples, ans2idx):
    """
    Remove the answers that don't appear in our answer set.
    --------------------
    Arguments:
        examples (list): the json data that contains all of answers in the dataset.
        ans2idx (dict): a set of considered answers.
    Return:
        examples (list): the processed json data which contains only answers in the answer set.
    """
    for ex in examples:
        ex["ans_score"] = [list(filter(lambda x: prep_ans(x[0]) in ans2idx, answers)) for answers in ex["ans_score"]]
        # ex["mc"] = [list(filter(lambda x: prep_ans(x[0]) in ans2idx, answers)) for answers in ex["mc"]]

    return examples


def refine_data_mc(dataset):
    for index in range(len(dataset)):
        dataset[index]['ques_id'] = dataset[index]['ques_id'][0]
        dataset[index]['ques'] = dataset[index]['ques'][0]
        dataset[index]['mc'] = dataset[index]['mc'][0]
        if 'mc_ans' in dataset[index].keys():
            dataset[index]['mc_ans'] = dataset[index]['mc_ans'][0]
            dataset[index]['ans_score'] = dataset[index]['ans_score'][0]
            dataset[index]['ans_freq'] = dataset[index]['ans_freq'][0]
    return dataset


# Use once at the first to generate the answer dict and remove the used answers from the dataset
def build_ans_dict_mc():
    # build the answer dict from the train-val split
    print("processing the the answer_dict....")
    trainval_set = json.load(open("/ExpData/gwy/vqa/v1/vqa_train_val_MC.json", "r"))
    top_answers = get_top_answers(trainval_set, 8)
    # num_ans = len(top_answers)
    ans2idx = {}
    for idx, ans in enumerate(top_answers):
        ans2idx[ans] = idx
    idx2ans = top_answers

    print("processing the trainval split....")
    trainval_set = filter_answers_mc(trainval_set, ans2idx)
    trainval_set = refine_data_mc(trainval_set)
    with open("/ExpData/gwy/vqa/v1/preprocessed/vqa_train_val_MC.json", "w+") as file:
        json.dump(obj=trainval_set, fp=file)
    with open("/ExpData/gwy/vqa/v1/preprocessed/ans_dict_MC.json", "w+") as file:
        json.dump(obj=(ans2idx, idx2ans), fp=file)

    print("processing the train set....")
    train_set = json.load(open("/ExpData/gwy/vqa/v1/vqa_train_MC.json", "r"))
    train_set = filter_answers_mc(train_set, ans2idx)
    train_set = refine_data_mc(train_set)
    with open("/ExpData/gwy/vqa/v1/preprocessed/vqa_train_MC.json", "w+") as file:
        json.dump(obj=train_set, fp=file)

    print("Process val dataset...")
    val_set = json.load(open("/ExpData/gwy/vqa/v1/vqa_val_MC.json", "r"))
    val_set = filter_answers_mc(val_set, ans2idx)
    val_set = refine_data_mc(val_set)
    with open("/ExpData/gwy/vqa/v1/preprocessed/vqa_val_MC.json", "w+") as file:
        json.dump(obj=val_set, fp=file)

    print("process testdev split...")
    testdev_set = json.load(open("/ExpData/gwy/vqa/v1/vqa_test_MC.json", "r"))
    testdev_set = refine_data_mc(testdev_set)
    with open("/ExpData/gwy/vqa/v1/preprocessed/vqa_test_MC.json", "w+") as file:
        json.dump(testdev_set, fp=file)

    print("process test split...")
    test_set = json.load(open("/ExpData/gwy/vqa/v1/vqa_testdev_MC.json", "r"))
    test_set = refine_data_mc(test_set)
    with open("/ExpData/gwy/vqa/v1/preprocessed/vqa_testdev_MC.json", "w+") as file:
        json.dump(test_set, fp=file)


# build_ans_dict_mc()


def proc_ans_mc(ques, ans_to_ix, token_to_ix):
    answers = ques['ans_score']
    mcs = ques['mc']
    ans_gt = ques['mc_ans']

    ans_label = np.zeros(18, np.float32)
    ans_mc_ix = np.zeros((18, 4), np.int)

    for index in range(len(mcs)):
        if mcs[index].lower() == ans_gt.lower():
            ans_label[index] = 1.0

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            mcs[index].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ans_mc_ix[ix] = token_to_ix[word]
            else:
                ans_mc_ix[ix] = token_to_ix['UNK']

            if ix + 1 == 4:
                break

    assert sum(ans_label) == 1.0

    ans_ix = np.zeros(18, np.int)-1
    for index in range(len(mcs)):
        try:
            ans_ix[index] = ans_to_ix[mcs[index]]
        except:
            pass

    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
    for ans in answers:
        ans_proc = prep_ans(ans[0])
        ans_score[ans_to_ix[ans_proc]] = ans[1]

    return ans_score, ans_label, ans_mc_ix, ans_ix


def proc_ans_mc_test(ques, ans_to_ix, token_to_ix):
    mcs = ques['mc']

    ans_mc_ix = np.zeros((18, 4), np.int)

    for index in range(len(mcs)):
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            mcs[index].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ans_mc_ix[ix] = token_to_ix[word]
            else:
                ans_mc_ix[ix] = token_to_ix['UNK']

            if ix + 1 == 4:
                break

    ans_ix = np.zeros(18, np.int)-1
    for index in range(len(mcs)):
        try:
            ans_ix[index] = ans_to_ix[mcs[index]]
        except:
            pass

    return ans_mc_ix, ans_ix


class DataSetMC(Data.Dataset):

    def __init__(self, __C):
        self.__C = __C

        # Loading all image paths
        # if self.__C.PRELOAD:
        self.img_feat_path_list = []
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            if split in ['train', 'val', 'test']:
                self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH[split] + '*.npz')

        # Loading question word list
        self.stat_ques_list = \
            json.load(open(data_map_vqa_mc['train'], 'r')) + \
            json.load(open(data_map_vqa_mc['val'], 'r')) + \
            json.load(open(data_map_vqa_mc['test'], 'r')) + \
            json.load(open(data_map_vqa_mc['testdev'], 'r'))

        # Loading question and answer list
        self.ques_list = []
        # self.ans_list = []

        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(data_map_vqa_mc[split], 'r'))
            # if __C.RUN_MODE in ['train']:
            #     self.ans_list += json.load(open(data_map_vqa_oe[split], 'r'))['annotations']

        # Define run data size
        # if __C.RUN_MODE in ['train']:
        #     self.data_size = self.ans_list.__len__()
        # else:
        self.data_size = self.ques_list.__len__()

        print('== Dataset size:', self.data_size)

        # {image id} -> {image feature absolutely path}
        self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)

        # {question id} -> {question}
        self.qid_to_ques = ques_load(self.ques_list, element_name='ques_id')

        # Tokenize
        self.token_to_ix, self.pretrained_emb = self.tokenize_mc(self.stat_ques_list, __C.USE_GLOVE,
                                                                 element_name='ques')
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        self.ans_to_ix, self.ix_to_ans = ans_stat('/ExpData/gwy/vqa/v1/preprocessed/ans_dict_MC.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')

    def tokenize_mc(self, stat_ques_list, use_glove, element_name='question'):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)

        for ques in stat_ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques[element_name].lower()
            ).replace('-', ' ').replace('/', ' ').split()

            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

            for answer in ques['mc']:
                words = re.sub(
                    r"([.,'!?\"()*#:;])",
                    '',
                    answer.lower()
                ).replace('-', ' ').replace('/', ' ').split()

                for word in words:
                    if word not in token_to_ix:
                        token_to_ix[word] = len(token_to_ix)
                        if use_glove:
                            pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb

    def __getitem__(self, idx):

        # For code safety
        pad = np.zeros(1)
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)

        # Process ['train'] and ['val', 'test'] respectively
        if self.__C.RUN_MODE in ['train']:
            # Load the run data from list
            ques = self.ques_list[idx]

            # CHANGED
            try:
                img_feat = np.load(self.iid_to_img_feat_path[str(ques['img_id'])])
                img_feat_x = img_feat['x'].transpose((1, 0))
                bboxes = img_feat['bbox']
                img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)
            except:
                print('false')
                print(self.iid_to_img_feat_path[str(ques['img_id'])])

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN, element_name='ques')

            # Process answer ans_score, ans_label, ans_mc_ix, ans_gt_ix
            ans_iter, ans_label, ans_mc_ix, ans_ix = proc_ans_mc(ques, self.ans_to_ix, self.token_to_ix)
            # ans_score, ans_label, ans_mc_ix, ans_ix
            return torch.from_numpy(img_feat_iter), \
                   torch.from_numpy(ques_ix_iter), \
                   torch.from_numpy(ans_iter), \
                   ques['img_id'], \
                   pad, \
                   ques['ques_id'], \
                   torch.from_numpy(ans_label), \
                   torch.from_numpy(ans_mc_ix), \
                   ans_ix, \
                   ques['mc']

        else:
            # Load the run data from list
            ques = self.ques_list[idx]

            img_feat = np.load(self.iid_to_img_feat_path[str(ques['img_id'])])
            img_feat_x = img_feat['x'].transpose((1, 0))
            bboxes = img_feat['bbox']
            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN, element_name='ques')

            ans_mc_ix, ans_ix = proc_ans_mc_test(ques, self.ans_to_ix, self.token_to_ix)

            return torch.from_numpy(img_feat_iter), \
                   torch.from_numpy(ques_ix_iter), \
                   pad, \
                   ques['img_id'], \
                   pad, \
                   ques['ques_id'], \
                   pad, \
                   torch.from_numpy(ans_mc_ix), \
                   ans_ix, \
                   ques['mc']

    def __len__(self):
        return self.data_size
