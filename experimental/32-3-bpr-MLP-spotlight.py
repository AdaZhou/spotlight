from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import mrr_score
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.interactions import Interactions
import pandas as pd
from spotlight.evaluation import *
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
from spotlight.factorization.representations import *
import os

input_config = os.environ

if 'SUFFIX' in input_config:
    suffix = input_config['SUFFIX']
else:
    suffix = "1"

if 'LOSS' in input_config:
    loss = input_config['LOSS']
else:
    loss="bpr"

if 'LR' in input_config:
    lr = float(input_config['LR'])
else:
    lr=1e-3

if 'L2' in input_config:
    l2 = float(input_config['L2'])
else:
    l2=1e-5
    
if 'MOM' in input_config:
    mom = float(input_config['MOM'])
else:
    mom=0.9

if 'SAMPLE' in input_config:
    train_sample = float(input_config['SAMPLE'])
else:
    train_sample = 1.0

if 'NEGSAMPLES' in input_config:
    num_negative_samples = int(input_config['NEGSAMPLES'])
else:
    num_negative_samples = 5


if 'BATCH' in input_config:
    batch_size = int(input_config['BATCH'])
else:
    batch_size=1024

dropout = None
if 'DROPOUT' in input_config and input_config['DROPOUT']:
    dropout = float(input_config['DROPOUT'])

net_conf = "32-3-bpr-MLP"
if 'NETCONG' in input_config and input_config['NETCONG']:
    net_conf = input_config['NETCONG']

amsgrad = False
if 'AMSGRAD' in input_config:
    amsgrad = (input_config['AMSGRAD'] == 'True')

adamw = None
if 'ADAMW' in input_config and input_config['ADAMW']:
    adamw=input_config['ADAMW']

betas=(mom, 0.999)
use_cuda=True

tensorboard_base_dir="runs"
model_alias = "{}-suf-{}-loss-{}-lr-{}-l2-{}-mom-{}-smpl-{}-ng-{}-bt-{}-drp-{}-amsgrad-{}-adamw={}".format(net_conf, suffix, loss, lr, l2, 
                                                                                      mom, train_sample,
                                                                                      num_negative_samples,
                                                                                     batch_size,
                                                                                                    dropout,
                                                                                      amsgrad,
                                                                                      adamw)
model_store_dir="models"
n_iters=100
#loss="adaptive_hinge"

log_loss_interval=1000
log_eval_interval=20000
#train_data_path = "s3a://tubi-playground-production/smistry/emb3/train-aug-28-phase1"
train_data_path = "data/train-aug-28-phase" + suffix

logging.basicConfig(filename="logs/" + model_alias + '.log', 
                    filemode='w', 
                    format='%(asctime)s - %(message)s',
                   level=logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

original_train_data = pd.read_parquet(train_data_path)
logger.info("Data is loaded")
writer = SummaryWriter(log_dir='{}/{}'.format(tensorboard_base_dir, model_alias))
writer.add_text('alias', model_alias, 0)

def notify_loss_completion(epoch_id, batch_id, loss, net, model):
    #print("notify_loss_completion")
    writer.add_scalar("Batch/loss", loss, batch_id)
    logging.info('[Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))

def notify_batch_eval_completion(epoch_id, batch_id, loss, net, model):
    #print("notify_batch_eval_completion")
    pairs_ndcg = nn_pairs_ndcg_score(net)
    writer.add_scalar("Batch/pairs_ndcg", pairs_ndcg, batch_id)
    logging.info('[Epoch {}] Batch {}, Embs NDCG = {:.4f}'.format(epoch_id, batch_id, pairs_ndcg))
    
def notify_epoch_completion(epoch_num, total_loss, net, model):
    #print("notify_epoch_completion")
    writer.add_scalar("Epoch/loss", total_loss, epoch_num)
    pairs_ndcg = nn_pairs_ndcg_score(net)
    writer.add_scalar("Epoch/pairs_ndcg", pairs_ndcg, epoch_num)
#     hit_ratio, ndcg = evaluate_hit_ratio_and_ndcg(model)
#     writer.add_scalar("Epoch/HR", hit_ratio, epoch_num)
#     writer.add_scalar("Epoch/NDCG", ndcg, epoch_num)
    hit_ratio, ndcg = -1,-1
    logging.info('******** [Epoch {}]  Embs NDCG {:.4f}, Hit Ratio: {:.4f}, NDCG: {:.4f}'.format(epoch_num,
                                                                                                    pairs_ndcg,
                                                                                                    hit_ratio,
                                                                                                    ndcg))
    torch.save(net, model_store_dir + "/" + model_alias + "-" + str(epoch_num))
    
num_users=len(original_train_data["uindex"].unique())
num_items=len(original_train_data["vindex"].unique())

train_data = original_train_data.sample(frac=train_sample)

interactions = Interactions(train_data["uindex"].to_numpy(),
            train_data["vindex"].to_numpy(),
            train_data["pct_cvt"].to_numpy(),
            train_data["latest_watch_time"].to_numpy(),
            num_users=len(original_train_data["uindex"].unique()),
            num_items=len(original_train_data["vindex"].unique()))

if "-" in net_conf:
    args = net_conf.split("-")
    config = {
          "factor_size": int(args[0]),
          "num_layers": int(args[1]),
          "loss_type": args[2],
          "model_type": args[3],
        "num_users": num_users,
        "num_items": num_items,
    }
    if dropout:
        config["dropout"] = dropout
    
    num_layers = int(args[1])
    factor_size = int(args[0])
    config["layers"] = [4 * factor_size] + [factor_size * (2 ** i) for i in range(num_layers - 1, -1, -1)]
    config["latent_dim"] = 2 * factor_size
    writer.add_text('config', str(config), 0)
    
    rep = MLP(config)
else:
    rep = None

model = ImplicitFactorizationModel(n_iter=n_iters,
                                   loss=loss,
                                  notify_loss_completion=notify_loss_completion,
                                  notify_batch_eval_completion=notify_batch_eval_completion,
                                  notify_epoch_completion=notify_epoch_completion,
                                  log_loss_interval=log_loss_interval,
                                  log_eval_interval=log_eval_interval,
                                  betas=betas,
                                  learning_rate=lr,
                                  batch_size=batch_size,
                                  random_state=np.random.RandomState(2),
                                  num_negative_samples=num_negative_samples,
                                  l2=l2,
                                  use_cuda=use_cuda,
                                  representation=rep,
                                  amsgrad=amsgrad,
                                 adamw=adamw)
logger.info("Model is initialized, now fitting..")
model.fit(interactions)


