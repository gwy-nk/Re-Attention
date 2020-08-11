from cfgs.base_cfgs import Cfgs
import argparse, yaml
import numpy as np
import torch

torch.backends.cudnn.enabled = False


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='MCAN Args')

    parser.add_argument('--recon_rate', dest='recon_rate',
                        default='1', type=float,
                        # required=True
                        )

    parser.add_argument('--entropy_tho', dest='entropy_tho',
                        default=0.1, type=float,
                        # required=True
                        )

    parser.add_argument('--AGCAN_MODE', dest='model_type',
                        choices=['recon', 'recon_e'],
                        type=str,
                        # required=True
                        default='recon_e',
                        )

    parser.add_argument('--RUN', dest='RUN_MODE',
                        choices=['train', 'val', 'test'],
                        help='{train, val, test}',
                        default='test',
                        type=str, required=False)

    parser.add_argument('--MODEL', dest='MODEL',
                        choices=['small', 'large'],
                        help='{small, large}',
                        default='small', type=str)

    parser.add_argument('--SPLIT', dest='TRAIN_SPLIT',
                        choices=['train', 'train+val', 'train+val+vg'],
                        help="set training split, "
                             "eg.'train', 'train+val+vg'"
                             "set 'train' can trigger the "
                             "eval after every epoch",
                        default='train',
                        type=str)

    parser.add_argument('--EVAL_EE', dest='EVAL_EVERY_EPOCH',
                        help='set True to evaluate the '
                             'val split when an epoch finished'
                             "(only work when train with "
                             "'train' split)",
                        type=bool)

    parser.add_argument('--SAVE_PRED', dest='TEST_SAVE_PRED',
                        help='set True to save the '
                             'prediction vectors'
                             '(only work in testing)',
                        type=bool)

    parser.add_argument('--BS', dest='BATCH_SIZE',
                        help='batch size during training',
                        type=int)

    parser.add_argument('--MAX_EPOCH', dest='MAX_EPOCH',
                        help='max training epoch',
                        type=int)

    parser.add_argument('--PRE_E', dest='PRE_E',
                        help='max training epoch',
                        type=int)

    parser.add_argument('--PRELOAD', dest='PRELOAD',
                        help='pre-load the features into memory'
                             'to increase the I/O speed',
                        type=bool)

    # changed
    parser.add_argument('--GPU', dest='GPU',
                        help="gpu select, eg.'0, 1, 2'",
                        default='1',
                        type=str)

    parser.add_argument('--SEED', dest='SEED',
                        help='fix random seed',
                        type=int)

    # changed recon_final
    parser.add_argument('--VERSION', dest='VERSION',
                        help='version control',
                        default='tmp',
                        type=str)

    parser.add_argument('--RESUME', dest='RESUME',
                        help='resume training',
                        default=True,
                        type=bool)

    parser.add_argument('--CKPT_V', dest='CKPT_VERSION',
                        help='checkpoint version',
                        type=str)

    parser.add_argument('--CKPT_E', dest='CKPT_EPOCH',
                        help='checkpoint epoch',
                        type=int)

    parser.add_argument('--CKPT_PATH', dest='CKPT_PATH',
                        help='load checkpoint path, we '
                             'recommend that you use '
                             'CKPT_VERSION and CKPT_EPOCH '
                             'instead',
                        type=str,
                        default='/home/gwy/code/vqa-journal/ckpts/ckpt_VQA_MC_base_bi_recon_e_2_10/epoch15.pkl')

    parser.add_argument('--ACCU', dest='GRAD_ACCU_STEPS',
                        help='reduce gpu memory usage',
                        type=int)

    parser.add_argument('--NW', dest='NUM_WORKERS',
                        help='multithreaded loading',
                        type=int)

    parser.add_argument('--PINM', dest='PIN_MEM',
                        help='use pin memory',
                        type=bool)

    parser.add_argument('--VERB', dest='VERBOSE',
                        help='verbose print',
                        type=bool)

    parser.add_argument('--DATA_PATH', dest='DATASET_PATH',
                        help='vqav2 dataset root path',
                        type=str)

    parser.add_argument('--FEAT_PATH', dest='FEATURE_PATH',
                        help='bottom up features root path',
                        type=str)

    parser.add_argument('--DATASET', dest='DATASET',
                        help='dataset and path',
                        default='VQA1+MC',
                        type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    __C = Cfgs()

    args = parse_args()
    args_dict = __C.parse_to_dict(args)

    cfg_file = "cfgs/{}_model.yml".format(args.MODEL)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f)

    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(args_dict)
    __C.proc()

    print('Hyper Parameters:')
    print(__C)

    __C.check_path()

    if args_dict['DATASET'] == "VQA2":
        from core.exec_VQA2 import Execution
    elif args_dict['DATASET'] in ['VQA1+OE', 'VQA1+MC']:
        from core.exec_VQA import Execution
    else:
        from core.exec_COCOQA import Execution

    execution = Execution(__C)
    print(__C.RUN_MODE)
    execution.run(__C.RUN_MODE)
