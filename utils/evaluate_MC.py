import os
import argparse
import logging
import json
import json
import os
import random
import scipy.io
from collections import defaultdict


# modified from Karpathy's neuraltalk (https://github.com/karpathy/neuraltalk)
class BasicDataProvider:

    def __init__(self, dataset, **kwargs):
        print('Initializing data provider for dataset %s...' % (dataset,))

        # load the dataset into memory
        dataset_root = "/home/gwy/code/vqa-journal/ICCV19_VQA-CTI-master/data_v7w/raw"
        dataset_path = os.path.join(dataset_root, 'dataset.json')
        print('BasicDataProvider: reading %s' % (dataset_path,))
        self.dataset = json.load(open(dataset_path, 'r'))

        # group images by their train/val/test split into a dictionary -> list structure
        self.split = defaultdict(list)
        for img in self.dataset['images']:
            self.split[img['split']].append(img)

        # load anwer boxes when available (for pointing QA)
        if 'boxes' in self.dataset:
            # load object bounding boxes
            self.grounding_boxes = dict()
            for box in self.dataset['boxes']:
                self.grounding_boxes[box['box_id']] = box

    # "PRIVATE" FUNCTIONS
    # in future we may want to create copies here so that we don't touch the
    # data provider class data, but for now lets do the simple thing and
    # just return raw internal img qa pair structs. This also has the advantage
    # that the driver could store various useful caching stuff in these structs
    # and they will be returned in the future with the cache present
    def _getImage(self, img):
        """ create an image structure """
        return img

    def _getQAPair(self, qa_pair):
        """ create a QA pair structure """
        return qa_pair

    def _getGroundingBox(self, box_id):
        """ create an answer box structure """
        return self.grounding_boxes[box_id]

    def _getQAMultipleChoice(self, qa_pair, shuffle=False):
        """ create a QA multiple choice structure """
        qa_pair = self._getQAPair(qa_pair)
        if 'multiple_choices' in qa_pair:
            mcs = qa_pair['multiple_choices']
            pos_idx = range(len(mcs) + 1)
            # random shuffle the positions of multiple choices
            if shuffle: random.shuffle(pos_idx)
            qa_pair['mc_candidates'] = []
            for idx, k in enumerate(pos_idx):
                if k == 0:
                    qa_pair['mc_candidates'].append(qa_pair['answer'])
                    qa_pair['mc_selection'] = idx  # record the position of the true answer
                else:
                    qa_pair['mc_candidates'].append(mcs[k - 1])
        return qa_pair

    # PUBLIC FUNCTIONS
    def getSplitSize(self, split, ofwhat='qa_pairs'):
        """ return size of a split, either number of QA pairs or number of images """
        if ofwhat == 'qa_pairs':
            return sum(len(img['qa_pairs']) for img in self.split[split])
        else:  # assume images
            return len(self.split[split])

    def sampleImageQAPair(self, split='train'):
        """ sample image QA pair from a split """
        images = self.split[split]

        img = random.choice(images)
        pair = random.choice(img['qa_pairs'])

        out = {}
        out['image'] = self._getImage(img)
        out['qa_pair'] = self._getQAPair(pair)
        return out

    def sampleImageQAMultipleChoice(self, split='train', shuffle=False):
        """ sample image and a multiple-choice test from a split """
        images = self.split[split]

        img = random.choice(images)
        pair = random.choice(img['qa_pairs'])

        out = {}
        out['image'] = self._getImage(img)
        out['mc'] = self._getQAMultipleChoice(pair, shuffle)
        return out

    def iterImageQAPair(self, split='train', max_images=-1):
        for i, img in enumerate(self.split[split]):
            if max_images >= 0 and i >= max_images: break
            for pair in img['qa_pairs']:
                out = {}
                out['image'] = self._getImage(img)
                out['qa_pair'] = self._getQAPair(pair)
                yield out

    def iterImageQAMultipleChoice(self, split='train', max_images=-1, max_batch_size=100, shuffle=False):
        for i, img in enumerate(self.split[split]):
            if max_images >= 0 and i >= max_images: break
            for pair in img['qa_pairs']:
                out = {}
                out['image'] = self._getImage(img)
                out['mc'] = self._getQAMultipleChoice(pair, shuffle)
                yield out

    def iterImageQAPairBatch(self, split='train', max_images=-1, max_batch_size=100):
        batch = []
        for i, img in enumerate(self.split[split]):
            if max_images >= 0 and i >= max_images: break
            for pair in img['qa_pairs']:
                out = {}
                out['image'] = self._getImage(img)
                out['qa_pair'] = self._getQAPair(pair)
                batch.append(out)
                if len(batch) >= max_batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def iterQAMultipleChoice(self, split='train', shuffle=False):
        for img in self.split[split]:
            for pair in img['qa_pairs']:
                yield self._getQAMultipleChoice(pair, shuffle)

    def iterQAPairs(self, split='train'):
        for img in self.split[split]:
            for pair in img['qa_pairs']:
                yield self._getQAPair(pair)

    def iterImages(self, split='train', shuffle=False, max_images=-1):
        imglist = self.split[split]
        ix = range(len(imglist))
        if shuffle:
            random.shuffle(ix)
        if max_images > 0:
            ix = ix[:min(len(ix), max_images)]  # crop the list
        for i in ix:
            yield self._getImage(imglist[i])

    def iterGroundingBoxes(self, shuffle=False, max_boxes=-1):
        if hasattr(self, 'grounding_boxes'):
            ix = self.grounding_boxes.keys()
            if shuffle:
                random.shuffle(ix)
            if max_boxes > 0:
                ix = ix[:min(len(ix), max_boxes)]
            for i in ix:
                yield self._getGroundingBox(i)


def getDataProvider(dataset, **kwargs):
    """ we could intercept a special dataset and return different data providers """
    assert dataset in ['visual7w-telling', 'visual7w-pointing'], 'dataset %s unknown' % (dataset,)
    return BasicDataProvider(dataset, **kwargs)

"""
  Compare top K candidate predictions with ground-truth answers
  We say that a model predicts the correct answer if one of the
  top k predictions match exactly with the ground-truth answers.
  Accuracy is used to report the performance.

  When evaluating multiple-choice QAs, the model makes a single
  prediction (i.e., the multiple-choice option it selects).

  - dp: data provider (an access helper to QA dataset)
  - params: evaluation mode configurations
"""
def evaluate_top_k(dp, params):
  # set parameter
  top_k = params['topk']
  if params['mode'] == 'mc':
    logging.info('Multiple-choice QA evaluation')
    if top_k != 1:
      logging.info('top_k is set to 1 for multiple-choice QA')
      top_k = 1
  else:
    logging.info('Open-ended QA evaluation')

  # split to be evaluated
  split = params['split']
  if split not in ['train', 'val', 'test']:
    logging.error('Error: cannot find split %s.' % split)
    return

  # load result json
  result_file = params['results']
  try:
    results = json.load(open(result_file))
  except:
    logging.error('Error: cannot read result file from %s' % result_file)
    return
  
  # initialize counters
  num_correct = 0
  num_total = 0
  
  # fetch all test QA pairs from data provider
  pairs = {pair['qa_id']: pair for pair in dp.iterQAPairs(split)}
  
  # record performances per question category
  category_total = dict()
  category_correct = dict()
  
  # loop through each prediction and check with ground-truth
  for idx, entry in enumerate(results):
    if entry['qa_id'] not in pairs:
      logging.error('Cannot find QA #%d. Are you using the correct split?' % entry['qa_id'])
      return
    pair = pairs[entry['qa_id']]
    answer = str(pair['answer']).lower()
    candidates = entry['candidates'][:top_k]
    c = pair['type']
    category_total[c] = category_total.get(c, 0) + 1
    for candidate in candidates:
      prediction = str(candidate['answer']).lower()
      if prediction == answer:
        num_correct += 1
        category_correct[c] = category_correct.get(c, 0) + 1
        break
    num_total += 1
    if (idx+1) % 10000 == 0:
      logging.info('Evaluated %s QA pairs...' % format(idx+1, ',d'))

  logging.info('Done!\n')
  logging.info('Evaluated on %s QA pairs with top-%d predictions.' % (format(num_total, ',d'), top_k))

  # # compute sub accuracy
  # for key, value in category_total.items():
  #   logging.info('Accuracy on %s is %.3f.' % (key, 1.0 * category_correct[key] / value))

  # compute metrics
  accuracy = 1.0 * num_correct / num_total
  logging.info('Overall accuracy = %.3f' % accuracy)

  verbose = params['verbose']
  if verbose:
      for c in category_total.keys():
          total = category_total.get(c, 0)
          correct = category_correct.get(c, 0)
          logging.info('Question type "%s" accuracy = %.3f (%d / %d)' % (c, 1.0 * correct / total, correct, total))

  return accuracy, category_total, category_correct


def evaluate_MC(split, result_file_name):
    # configure logging settings
    FORMAT = "%(asctime)-15s %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    paras = {
            "dataset": 'visual7w-telling',
            "mode": 'mc',
            "topk": 1,
            "results": result_file_name,
            "split": split,
            "verbose": 1
    }

    dp = getDataProvider(paras['dataset'])

    # start evaluation mode
    # if paras['dataset'].endswith('telling'):
    #   # multiple-choice and open-ended evaluations are supported in telling QA
    #   assert paras['mode'] in ['mc', 'open'], 'Evaluation mode %s not supported in telling QA.' % params['mode']
    accuracy, category_total, category_correct = evaluate_top_k(dp, paras)
    return accuracy, category_total, category_correct

# if __name__ == '__main__':
#
#   # configure logging settings
#   FORMAT = "%(asctime)-15s %(message)s"
#   logging.basicConfig(format=FORMAT, level=logging.DEBUG)
#
#   # configure argument parser
#   parser = argparse.ArgumentParser()
#   parser.add_argument('-d', '--dataset', default='visual7w-telling', type=str, help='dataset name (default: visual7w-telling)')
#   parser.add_argument('-m', '--mode', default='mc', type=str, help='prediction mode: "mc" - multiple-choice QA; "open" - open-ended QA.')
#   parser.add_argument('-k', '--topk', default=1, type=int, help='top-k evaluation. k is the number of answer candidates to be examined.')
#   parser.add_argument('-j', '--results', default='/home/gwy/code/vqa-journal/ICCV19_VQA-CTI-master/src/MC/v7w_result.json', help='path to json file contains the results')
#   parser.add_argument('-s', '--split', type=str, default='test', help='the split to be evaluated: train / val / test (default: val)')
#   parser.add_argument('-v', '--verbose', default=0, type=int, help='verbose mode. report performances of question categories when enabled.')
#
#   # parse arguments
#   args = parser.parse_args()
#   params = vars(args) # convert to ordinary dict
#
#   # load dataset (skipping feature files)
#   dp = getDataProvider(params['dataset'])
#
#   # start evaluation mode
#   if params['dataset'].endswith('telling'):
#     # multiple-choice and open-ended evaluations are supported in telling QA
#     assert params['mode'] in ['mc', 'open'], 'Evaluation mode %s not supported in telling QA.' % params['mode']
#     evaluate_top_k(dp, params)

# evaluate_MC('val', "v7w_result.json")
