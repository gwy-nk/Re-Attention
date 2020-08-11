from core.data.data_utils import img_feat_load, ques_load, tokenize, ans_stat  #img_feat_path_load
from core.data.data_utils import proc_img_feat, proc_ques, proc_ans

import numpy as np
import glob, json, torch, time
import torch.utils.data as Data
import pickle as pkl


def img_feat_path_load(path_list):
    iid_to_path = {}

    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        iid_to_path[iid] = path

    return iid_to_path


class DataSet(Data.Dataset):

    def __init__(self, __C):
        self.__C = __C
        self.QUESTION_PATH = {
            'train': "/home/gwy/code/vqa-journal/datasets/coco-qa/coco_qa_train.json",
            'val': "/home/gwy/code/vqa-journal/datasets/coco-qa/coco_qa_test.json",
            'test': "/home/gwy/code/vqa-journal/datasets/coco-qa/coco_qa_test.json"
        }
        # self.ANSWER_PATH = {
        #     'train': "/home/gwy/code/vqa-journal/datasets/TDIUC/mscoco_train2014_annotations.json",
        #     'val': "/home/gwy/code/vqa-journal/datasets/TDIUC/mscoco_val2014_annotations.json"
        # }

        # self.VG_IMG_PATH = {
        #     'train': "/ExpData/gwy/Visual_Genome/TDIUC_train2014/",
        #     'val': "/ExpData/gwy/Visual_Genome/TDIUC_val2014/"
        # }

        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        # Loading all image paths
        # if self.__C.PRELOAD:
        self.img_feat_path_list = []
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            if split in ['train', 'val']:
                self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH[split] + '*.npz')
                # self.img_feat_path_list += glob.glob(self.VG_IMG_PATH[split] + '*.pkl')
            elif split in ['test']:
                self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH['val'] + '*.npz')

        # Loading question word list
        self.stat_ques_list = \
            json.load(open(self.QUESTION_PATH['train'], 'r'))['questions'] + \
            json.load(open(self.QUESTION_PATH['val'], 'r'))['questions']
            # json.load(open(__C.QUESTION_PATH['test'], 'r'))['questions'] + \
            # json.load(open(__C.QUESTION_PATH['vg'], 'r'))['questions']

        # Loading question and answer list
        self.ques_list = []
        # self.ans_list = []

        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(self.QUESTION_PATH[split], 'r'))['questions']
            # if __C.RUN_MODE in ['train']:
            #     self.ans_list += json.load(open(self.ANSWER_PATH[split], 'r'))['annotations']

        # Define run data size
        # if __C.RUN_MODE in ['train']:
        #     self.data_size = self.ans_list.__len__()
        # else:
        self.data_size = self.ques_list.__len__()

        print('== Dataset size:', self.data_size)

        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)

        # {question id} -> {question}
        self.qid_to_ques = ques_load(self.ques_list)

        # Tokenize
        self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        # Answers statistic
        # Make answer dict during training does not guarantee
        # the same order of {ans_to_ix}, so we published our
        # answer dict to ensure that our pre-trained model
        # can be adapted on each machine.

        # Thanks to Licheng Yu (https://github.com/lichengunc)
        # for finding this bug and providing the solutions.

        # self.ans_to_ix, self.ix_to_ans = ans_stat(self.stat_ans_list, __C.ANS_FREQ)
        self.ans_to_ix, self.ix_to_ans = ans_stat('/home/gwy/code/vqa-journal/datasets/coco-qa/ans_dict_coco_qa.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')

        # self.VG_wrong_iids = [2344368, 2332085, 2330534, 2358818, 2371078, 2376567, 2326416, 2393350, 4385,
        #                       2394918, 2405199, 2397416, 2398959, 2415446]

    def __getitem__(self, idx):

        # For code safety
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)
        img_feat_x = np.zeros((1, 2048), dtype=np.float32)

        # Process ['train'] and ['val', 'test'] respectively
        if self.__C.RUN_MODE in ['train']:
            # Load the run data from list
            q_tmp = self.ques_list[idx]
            ques = self.qid_to_ques[str(q_tmp['question_id'])]

            # Process image feature from (.npz) file
            # if self.__C.PRELOAD:
            #     img_feat_x = self.iid_to_img_feat[str(ans['image_id'])]
            # else:
            try:
                img_feat = np.load(self.iid_to_img_feat_path[str(q_tmp['image_id'])])
                img_feat_x = img_feat['x'].transpose((1, 0))
            except:
                print("   , ", q_tmp['image_id'], "====")
                pass
                # except:
                #     print('false')
                #     print(self.iid_to_img_feat_path[str(ans['image_id'])])
            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

            # Process answer
            # ans_iter = proc_ans(ans, self.ans_to_ix)
            ans_iter = np.zeros(len(self.ans_to_ix), dtype=np.float32)
            ans_iter[self.ans_to_ix[ques['answer']]] = 1.0
            #
            # if sum(ans_iter==0.3) == 1:
            #     ans_iter[ans_iter == 0.3] = 1

        else:
            # Load the run data from list
            ques = self.ques_list[idx]

            # # Process image feature from (.npz) file
            # img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
            # img_feat_x = img_feat['x'].transpose((1, 0))
            # Process image feature from (.npz) file
            # if self.__C.PRELOAD:
            #     img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]
            # else:

            try:
                img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
                img_feat_x = img_feat['x'].transpose((1, 0))
            except:
                print(ques['image_id'], "============")
            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

            ans_iter = np.zeros(1, dtype=np.float32)
            ans_iter[0] = ques['question_id']

        return torch.from_numpy(img_feat_iter), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter)

    def __len__(self):
        return self.data_size


