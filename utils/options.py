# import argparse
import os
import glob

# def define_arguments(parser):
#     parser.add_argument('--json-root', type=str, default="/kaggle/working", help="")
#     parser.add_argument('--feature-root', type=str, default="/kaggle/input/sharpseq-features", help="")
#     parser.add_argument('--stream-file', type=str, default="/kaggle/working/sharpseq/data/MAVEN/streams.json", help="")
#     parser.add_argument('--batch-size', type=int, default=32, help="")
#     parser.add_argument('--init-slots', type=int, default=13, help="")
#     parser.add_argument('--patience', type=int, default=10, help="")
#     parser.add_argument('--input-dim', type=int, default=2048, help="")
#     parser.add_argument('--hidden-dim', type=int, default=512, help="")
#     parser.add_argument('--max-slots', type=int, default=169, help="")
#     parser.add_argument('--perm-id', type=int, default=0, help="")
#     parser.add_argument('--no-gpu', action="store_true", help="don't use gpu")
#     parser.add_argument('--gpu', type=int, default=0, help="gpu")
#     parser.add_argument('--learning-rate', type=float, default=1e-4, help="")
#     parser.add_argument('--decay', type=float, default=1e-2, help="")
#     parser.add_argument('--kt-alpha', type=float, default=0.25, help="")
#     parser.add_argument('--kt-gamma', type=float, default=0.05, help="")
#     parser.add_argument('--kt-tau', type=float, default=1.0, help="")
#     parser.add_argument('--kt-delta', type=float, default=0.5, help="")
#     parser.add_argument('--seed', type=int, default=2147483647, help="random seed")
#     parser.add_argument('--save-model', type=str, default="model", help="path to save checkpoints")
#     parser.add_argument('--load-model', type=str, default="", help="path to saved checkpoint")
#     parser.add_argument('--log-dir', type=str, default="./log/", help="path to save log file")
#     parser.add_argument('--train-epoch', type=int, default=50, help='epochs to train')
#     parser.add_argument('--test-only', action="store_true", help='is testing')
#     parser.add_argument('--kt', action="store_true", help='')
#     parser.add_argument('--kt2', action="store_true", help='')
#     parser.add_argument('--finetune', action="store_true", help='')
#     parser.add_argument('--load-first', type=str, default="", help="path to saved checkpoint")
#     parser.add_argument('--skip-first', action="store_true", help='')
#     parser.add_argument('--load-second', type=str, default="", help="path to saved checkpoint")
#     parser.add_argument('--skip-second', action="store_true", help='')
#     parser.add_argument('--balance', choices=['icarl', 'eeil', 'bic', 'none', 'fd', 'mul', 'nod'], default="none")
#     parser.add_argument('--setting', choices=['classic', "new"], default="classic")
#     parser.add_argument('--mode', choices=["kmeans", "herding", "GMM"], type=str, default="herding", help='exemplar algorithm')
#     parser.add_argument('--kt_mode', choices=["kmeans", "herding", "GMM"], type=str, default="herding", help='KT')
#     parser.add_argument('--clusters', type=int, default=4, help='the number of clusters')
#     parser.add_argument('--generate',  action="store_true", help="")
#     parser.add_argument('--sample-size', type=int, default=2, help="the sample size of each label in the replay and generated sets")
#     parser.add_argument('--features-distill',  action="store_true",  help="whether do feature distillation (just distill span mlp output) or not")
#     parser.add_argument('--hyer-distill',  action="store_true",  help="whether do feature hyer-distillation or not")
#     parser.add_argument('--reduce-na',  action="store_true",  help="reduce number of negative instances")


#     parser.add_argument('--new-test-mode',  action="store_true",  help="") # still debunging, not used
#     parser.add_argument('--num_loss', type=int, default=4, help='epochs to train')
#     parser.add_argument('--mul_task', action="store_true", help='epochs to train')
#     parser.add_argument("--contrastive", action="store_true", help="contrastive loss")
#     parser.add_argument("--mul_distill", action="store_true")
#     parser.add_argument("--mul_task_type", type=str, choices=['NashMTL','PCGrad','IMTLG', 'MGDA'], default='NashMTL')
#     parser.add_argument("--naive_replay", action="store_true")


#     parser.add_argument("--debug", action="store_true", help="for debug")
#     parser.add_argument("--colab_viet", action="store_true", help="util for run on colab")

#     parser.add_argument('--datasetname',  type=str, choices=['MAVEN', 'ACE', 'ACE_lifelong'], default='MAVEN')
#     parser.add_argument("--center-ratio", type=int, default=1, help="The number points that near to the center")
#     parser.add_argument("--generate_ratio", type=int, default=20, help="The ratio between replay set and generated set")
#     parser.add_argument("--naloss_ratio", type=int, default=4, help="")
#     parser.add_argument("--dropout", type=str, choices=["normal", "adap", "fixed"], default="normal")
#     parser.add_argument("--p", type=float, default=0.5)
#     parser.add_argument("--num_sam_loss", type=int, default=2)
#     parser.add_argument("--sam", type=int, default=1, help="sam")
#     parser.add_argument("--ot", action="store_true", help="for debug")
#     parser.add_argument("--llm2vec", action="store_true", help="llm2vec")

class Config:
    def __init__(
        self,
        json_root="/kaggle/working",
        # feature_root="/kaggle/working/features",
        feature_root="/kaggle/input/sharpseq-features",
        stream_file="/kaggle/working/ACE/streams.json",
        batch_size=512,
        init_slots=13,
        patience=6,
        input_dim=2048,
        hidden_dim=512,
        max_slots=34,
        perm_id=0,
        no_gpu=False,
        gpu=0,
        learning_rate=1e-4,
        decay=1e-2,
        kt_alpha=0.25,
        kt_gamma=0.05,
        kt_tau=1.0,
        kt_delta=0.5,
        seed=2147483647,
        save_model="model",
        load_model="/kaggle/working/checkpoint",
        log_dir="/kaggle/working/log",
        train_epoch=15,
        test_only=False,
        kt=True,
        kt2=True,
        finetune=False,
        load_first="",
        skip_first=False,
        load_second="",
        skip_second=False,
        balance="none",
        setting="classic",
        mode="herding",
        kt_mode="herding",
        clusters=4,
        generate=True,
        sample_size=2,
        features_distill=False,
        hyer_distill=False,
        reduce_na=False,
        new_test_mode=False,
        num_loss=4,
        mul_task=True,
        contrastive=False,
        mul_distill=True,
        mul_task_type="NashMTL",
        extra_weight_loss = 1,
        naive_replay=False,
        debug=False,
        colab_viet=False,
        datasetname="ACE",
        center_ratio=1,
        generate_ratio=20,
        naloss_ratio=40,
        dropout="normal",
        p=0.2,
        num_sam_loss=2,
        sam=1,
        ot=False,
        llm2vec=False,
        user_wandb=True

    ):
        self.json_root = json_root
        self.feature_root = feature_root
        self.stream_file = stream_file
        self.batch_size = batch_size
        self.init_slots = init_slots
        self.patience = patience
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_slots = max_slots
        self.perm_id = perm_id
        self.no_gpu = no_gpu
        self.gpu = gpu
        self.learning_rate = learning_rate
        self.decay = decay
        self.kt_alpha = kt_alpha
        self.kt_gamma = kt_gamma
        self.kt_tau = kt_tau
        self.kt_delta = kt_delta
        self.seed = seed
        self.save_model = save_model
        self.load_model = load_model
        self.log_dir = log_dir
        self.train_epoch = train_epoch
        self.test_only = test_only
        self.kt = kt
        self.kt2 = kt2
        self.finetune = finetune
        self.load_first = load_first
        self.skip_first = skip_first
        self.load_second = load_second
        self.skip_second = skip_second
        self.balance = balance
        self.setting = setting
        self.mode = mode
        self.kt_mode = kt_mode
        self.clusters = clusters
        self.generate = generate
        self.sample_size = sample_size
        self.features_distill = features_distill
        self.hyer_distill = hyer_distill
        self.reduce_na = reduce_na
        self.new_test_mode = new_test_mode
        self.num_loss = num_loss
        self.mul_task = mul_task
        self.contrastive = contrastive
        self.mul_distill = mul_distill
        self.mul_task_type = mul_task_type
        self.extra_weight_loss = extra_weight_loss
        self.naive_replay = naive_replay
        self.debug = debug
        self.colab_viet = colab_viet
        self.datasetname = datasetname
        self.center_ratio = center_ratio
        self.generate_ratio = generate_ratio
        self.naloss_ratio = naloss_ratio
        self.dropout = dropout
        self.p = p
        self.num_sam_loss = num_sam_loss
        self.sam = sam  
        self.ot = ot
        self.llm2vec = llm2vec
        self.user_wandb = user_wandb

    def __repr__(self):
        return str(self.__dict__)


PERM = [[0, 1, 2, 3,4], [4, 3, 2, 1, 0], [0, 3, 1, 4, 2], [1, 2, 0, 3, 4], [3, 4, 0, 1, 2]]

import json
def parse_arguments():
    # parser = argparse.ArgumentParser()
    # define_arguments(parser)
    # args = parser.parse_args()
    args = Config(train_epoch=20)
    args.log = os.path.join(args.log_dir, "logfile.log")

    if (not args.test_only) and os.path.exists(args.log_dir):
        existing_logs = glob.glob(os.path.join(args.log_dir, "*"))
        for _t in existing_logs:
            if 'exp.log' not in _t and not _t.endswith('.py'):
                os.remove(_t)
    print('Dump name space')
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(f'{args.log_dir}/options.json', 'w') as f:
        print(f'{args.log_dir}/options.json')
        json.dump(vars(args), f, ensure_ascii=False)

    return args