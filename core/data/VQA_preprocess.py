
import torch
import numpy as np

from collections import Counter
from torchtext import vocab
from torchtext.vocab import GloVe

#%%

gloves = GloVe(name="840B", dim="300")

#%%

c = Counter()
for word in gloves.itos:
    c.update([word])

print(len(c))
v = vocab.Vocab(c, min_freq=1, specials=["<_>", "<unk>"], vectors=gloves)

#%%

idx2word = v.itos
word2idx = v.stoi
vectors = v.vectors

#%%

print(len(idx2word))
print(len(word2idx.keys()))
print(len(vectors))

print(type(idx2word))
print(type(word2idx))
print(vectors.size())

#%%

vocab_path = "/ExpData/gwy/vqa/v1/glove_840B.pt"

info = {
    "idx2word": idx2word,
    "word2idx": word2idx,
    "vectors": vectors,
}
torch.save(info, vocab_path)

# exit()



# %% md

# Preprocess VQA 2.0

# %%

import json
import os
from collections import Counter
from tqdm import tqdm

# %%
raw_root = "/ExpData/gwy/vqa/v1/"

train_anno_path = "mscoco_train2014_annotations.json"
val_anno_path = "mscoco_val2014_annotations.json"

train_ques_path = "OpenEnded_mscoco_train2014_questions.json"
val_ques_path = "OpenEnded_mscoco_val2014_questions.json"
testdev_ques_path = "OpenEnded_mscoco_test-dev2015_questions.json"
test_ques_path = "OpenEnded_mscoco_test2015_questions.json"

train_save_path_OE = "vqa_train_oe.json"
val_save_path_OE = "vqa_val_oe.json"
test_save_path_OE = "vqa_test_oe.json"
testdev_save_path_OE = "vqa_testdev_oe.json"


def num_to_score(num):
    if num > 3:
        return 1
    elif num == 3:
        return 0.9
    elif num == 2:
        return 0.6
    elif num == 1:
        return 0.3
    elif num == 0:
        return 0
    else:
        raise TypeError("Wrong type of number!")


def save_dataset(subtype, ques_path, anno_path=None):
    dataset = []
    imdir = "/ExpData/gwy/coco_extract/%s/COCO_%s_%012d.jpg.npz"
    ques_data = json.load(open(ques_path, "r"))
    anno_data = json.load(open(anno_path, "r")) if anno_path is not None else None

    for i in tqdm(range(len(ques_data["questions"]))):
        ques = ques_data["questions"][i]["question"]
        ques_id = ques_data["questions"][i]["question_id"]
        img_id = ques_data["questions"][i]["image_id"]
        image_path = imdir % (subtype, subtype, img_id)

        item = {"ques_id": [ques_id], "ques": [ques], "img_path": image_path, "img_id": img_id}

        if anno_path is not None:
            mc_ans = anno_data["annotations"][i]["multiple_choice_answer"]
            answers = Counter()
            for ans in anno_data["annotations"][i]["answers"]:
                answers.update([ans["answer"]])
            # answers = [(ans, num_to_score(num)) for ans, num in answers.items()]
            #
            # assert img_id == anno_data["annotations"][i]["image_id"], "Image index doesn't match!"
            # assert ques_id == anno_data["annotations"][i]["question_id"], "Question index doesn't match!"
            #
            # item["mc_ans"] = [mc_ans]
            # item["ans"] = [answers]

            answers_score = [(ans, num_to_score(num)) for ans, num in answers.items()]
            answers_freq = [(ans, num) for ans, num in answers.items()]

            assert img_id == anno_data["annotations"][i]["image_id"], "Image index doesn't match!"
            assert ques_id == anno_data["annotations"][i]["question_id"], "Question index doesn't match!"

            item["mc_ans"] = [mc_ans]
            item["ans_score"] = [answers_score]
            item['ans_freq'] = [answers_freq]
        dataset.append(item)
        # if (i + 1) % 1000 == 0:
        #     print("processing %i/%i" % (i, len(ques_data["questions"])))

    return dataset


# #############################
print("train")
trainset = save_dataset("train2014", os.path.join(raw_root, train_ques_path), os.path.join(raw_root, train_anno_path))
print(len(trainset))
num_samples = 1
for i in range(num_samples):
    print(trainset[i])
with open(os.path.join(raw_root, train_save_path_OE), "w") as f:
    json.dump(trainset, f)


# #############################
print("val")
valset = save_dataset("val2014", os.path.join(raw_root, val_ques_path), os.path.join(raw_root, val_anno_path))
print(len(valset))
num_samples = 1
for i in range(num_samples):
    print(valset[i])
with open(os.path.join(raw_root, val_save_path_OE), "w") as f:
    json.dump(valset, f)


trainval_path = "/ExpData/gwy/vqa/v1/vqa_train_val_OE.json"
trainvalset = trainset + valset
with open(trainval_path, "w") as f:
    json.dump(trainvalset, f)


# #############################
print("testdev")
testdevset = save_dataset("test2015", os.path.join(raw_root, testdev_ques_path))
# %%
print(len(testdevset))
num_samples = 1
for i in range(num_samples):
    print(testdevset[i])
with open(os.path.join(raw_root, testdev_save_path_OE), "w") as f:
    json.dump(testdevset, f)


# #############################
print("test")
testset = save_dataset("test2015", os.path.join(raw_root, test_ques_path))
print(len(testset))
num_samples = 1
for i in range(num_samples):
    print(testset[i])
with open(os.path.join(raw_root, test_save_path_OE), "w") as f:
    json.dump(testset, f)




# %% md

# Preprocess VQA 1.0

# %%



import json

from collections import Counter

# %%

train_anno_path = "/ExpData/gwy/vqa/v1/mscoco_train2014_annotations.json"
val_anno_path = "/ExpData/gwy/vqa/v1/mscoco_val2014_annotations.json"

train_ques_path = "/ExpData/gwy/vqa/v1/OpenEnded_mscoco_train2014_questions.json"
val_ques_path = "/ExpData/gwy/vqa/v1/OpenEnded_mscoco_val2014_questions.json"
testdev_ques_path = "/ExpData/gwy/vqa/v1/OpenEnded_mscoco_test-dev2015_questions.json"
test_ques_path = "/ExpData/gwy/vqa/v1/OpenEnded_mscoco_test2015_questions.json"

trainmc_ques_path = "/ExpData/gwy/vqa/v1/MultipleChoice_mscoco_train2014_questions.json"
valmc_ques_path = "/ExpData/gwy/vqa/v1/MultipleChoice_mscoco_val2014_questions.json"
testdevmc_ques_path = "/ExpData/gwy/vqa/v1/MultipleChoice_mscoco_test-dev2015_questions.json"
testmc_ques_path = "/ExpData/gwy/vqa/v1/MultipleChoice_mscoco_test2015_questions.json"


# %%

def num_to_score(num):
    if num > 3:
        return 1
    elif num == 3:
        return 0.9
    elif num == 2:
        return 0.6
    elif num == 1:
        return 0.3
    elif num == 0:
        return 0
    else:
        raise TypeError("Wrong type of number!")


# %%

def save_dataset(subtype, ques_path, mc_path, anno_path=None):
    dataset = []
    imdir = "/ExpData/gwy/vqa/v1/%s/COCO_%s_%012d.jpg.npz"
    ques_data = json.load(open(ques_path, "r"))
    mc_data = json.load(open(mc_path, "r"))
    anno_data = json.load(open(anno_path, "r")) if anno_path is not None else None

    for i in tqdm(range(len(ques_data["questions"]))):
        ques = ques_data["questions"][i]["question"]
        ques_id = ques_data["questions"][i]["question_id"]
        mc = mc_data["questions"][i]["multiple_choices"]
        img_id = ques_data["questions"][i]["image_id"]
        image_path = imdir % (subtype, subtype, img_id)

        # item = {"ques_id": [ques_id], "img_path": image_path, "ques": [ques], "id": img_id}
        item = {"ques_id": [ques_id], "ques": [ques], "img_path": image_path, "img_id": img_id, "mc": [mc]}

        if anno_path is not None:
            mc_ans = anno_data["annotations"][i]["multiple_choice_answer"]
            answers = Counter()
            for ans in anno_data["annotations"][i]["answers"]:
                answers.update([ans["answer"]])
            answers_score = [(ans, num_to_score(num)) for ans, num in answers.items()]
            answers_freq = [(ans, num) for ans, num in answers.items()]

            assert img_id == anno_data["annotations"][i]["image_id"], "Image index doesn't match!"
            assert ques_id == anno_data["annotations"][i]["question_id"], "Question index doesn't match!"

            item["mc_ans"] = [mc_ans]
            item["ans_score"] = [answers_score]
            item['ans_freq'] = [answers_freq]
        dataset.append(item)
        # if (i + 1) % 1000 == 0:
        #     print("processing %i/%i" % (i, len(ques_data["questions"])))

    return dataset


# %%
print("train_MC")
trainset = save_dataset("train2014", train_ques_path, trainmc_ques_path, train_anno_path)
print(len(trainset))
num_samples = 1
for i in range(num_samples):
    print(trainset[i])
train_path = "/ExpData/gwy/vqa/v1/vqa_train_MC.json"
with open(train_path, "w") as f:
    json.dump(trainset, f)


valset = save_dataset("val2014", val_ques_path, valmc_ques_path, val_anno_path)
print(len(valset))
num_samples = 1
for i in range(num_samples):
    print(valset[i])
val_path = "/ExpData/gwy/vqa/v1/vqa_val_MC.json"
with open(val_path, "w") as f:
    json.dump(valset, f)


trainval_path = "/ExpData/gwy/vqa/v1/vqa_train_val_MC.json"
trainvalset = trainset + valset
with open(trainval_path, "w") as f:
    json.dump(trainvalset, f)


testdevset = save_dataset("test2015", testdev_ques_path, testdevmc_ques_path)
print(len(testdevset))
num_samples = 1
for i in range(num_samples):
    print(testdevset[i])
testdev_path = "/ExpData/gwy/vqa/v1/vqa_testdev_MC.json"
with open(testdev_path, "w") as f:
    json.dump(testdevset, f)


testset = save_dataset("test2015", test_ques_path, testmc_ques_path)
print(len(testset))
num_samples = 1
for i in range(num_samples):
    print(testset[i])
test_path = "/ExpData/gwy/vqa/v1/vqa_test_MC.json"
with open(test_path, "w") as f:
    json.dump(testset, f)
