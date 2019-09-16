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
import pandas as pd
import numpy as np
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
import collections
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import CNNNet

input_config = os.environ

hyperparameters = {
    'loss': "adaptive_hinge",
    'batch': 256,
    'lr': 1e-3,
    'l2': 1e-06,
    'n_iter': 50,
    'emb_dim': 64,
    'type': 'mixture'
}
h = hyperparameters

if 'SUFFIX' in input_config:
    suffix = input_config['SUFFIX']
else:
    suffix = "1"

if 'LOSS' in input_config:
    h['loss'] = input_config['LOSS']

if 'LR' in input_config:
    h['lr'] = float(input_config['LR'])

if 'L2' in input_config:
    h['l2'] = float(input_config['L2'])

if 'MOM' in input_config:
    h['mom'] = float(input_config['MOM'])
else:
    h['mom'] = 0.9

if 'NEGSAMPLES' in input_config:
    h['neg'] = int(input_config['NEGSAMPLES'])
else:
    h['neg'] = 5

if 'BATCH' in input_config:
    h['batch'] = int(input_config['BATCH'])

h['amsgrad'] = False
if 'AMSGRAD' in input_config:
    h['amsgrad'] = (input_config['AMSGRAD'] == 'True')

h['adamw'] = False
if 'ADAMW' in input_config and input_config['ADAMW']:
    h['adamw'] = (input_config['ADAMW'] == 'True')

if 'EMBDIM' in input_config:
    h['emb_dim'] = int(input_config['EMBDIM'])

betas = (h['mom'], 0.999)
use_cuda = True

tensorboard_base_dir = "sruns"
model_store_dir = "smodels"
n_iters = 50
# loss="adaptive_hinge"

log_loss_interval = 1000
log_eval_interval = 20000

model_alias = ",".join([k + "=" + str(v) for k, v in collections.OrderedDict(h).items()])
model_alias = "mixture_" + model_alias

# train_data_path = "s3a://tubi-playground-production/smistry/emb3/train-aug-28-phase1"
train_data_path = "/home/ec2-user/emb3/data/train-aug-28-phase" + suffix

logging.basicConfig(filename="slogs/" + model_alias + '.log',
                    filemode='w',
                    format='%(asctime)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

original_train_data = pd.read_parquet(train_data_path)
logger.info("Data is downloaded")
# train_data = original_train_data
uvs = original_train_data.groupby("uindex")["vindex"].agg(list)
train_data = original_train_data[original_train_data.uindex.isin(uvs[uvs.apply(lambda x: len(x)) <= 50].index)]

logger.info("Filtered train data..")

train_data["vindex"] = train_data["vindex"] + 1
interactions = Interactions(train_data["uindex"].to_numpy(),
                            train_data["vindex"].to_numpy(),
                            train_data["pct_cvt"].to_numpy(),
                            train_data["latest_watch_time"].to_numpy(),
                            num_users=len(original_train_data["uindex"].unique()),
                            num_items=len(original_train_data["vindex"].unique()) + 2)

max_sequence_length = 50
min_sequence_length = 2
step_size = 1

if "1500K" in suffix:
    logger.info("Increasing step size and max_sequence_length")
    step_size = 2
    min_sequence_length = 2
    max_sequence_length = 50

train_seq = interactions.to_sequence(max_sequence_length=max_sequence_length,
                                     min_sequence_length=min_sequence_length,
                                     step_size=step_size)

logger.info("Data is loaded and converted to sequences..")

writer = SummaryWriter(log_dir='{}/{}'.format(tensorboard_base_dir, model_alias))
writer.add_text('alias', model_alias, 0)
writer.add_text('hyperparameters', str(h), 0)


def notify_loss_completion(epoch_id, batch_id, loss, net, model):
    # print("notify_loss_completion")
    writer.add_scalar("Batch/loss", loss, batch_id)
    logging.info('[Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))


def notify_batch_eval_completion(epoch_id, batch_id, loss, net, model):
    # print("notify_batch_eval_completion")
    m = 1


def notify_epoch_completion(epoch_num, total_loss, net, model):
    # print("notify_epoch_completion")
    writer.add_scalar("Epoch/loss", total_loss, epoch_num)
    pairs_ndcg = nn_pairs_ndcg_score(net)
    writer.add_scalar("Epoch/pairs_ndcg", pairs_ndcg, epoch_num)
    #     hit_ratio, ndcg = evaluate_hit_ratio_and_ndcg(model)
    #     writer.add_scalar("Epoch/HR", hit_ratio, epoch_num)
    #     writer.add_scalar("Epoch/NDCG", ndcg, epoch_num)
    hit_ratio, ndcg = -1, -1
    logging.info('******** [Epoch {}]  Embs NDCG {:.4f}, Hit Ratio: {:.4f}, NDCG: {:.4f}'.format(epoch_num,
                                                                                                 pairs_ndcg,
                                                                                                 hit_ratio,
                                                                                                 ndcg))
    torch.save(net, model_store_dir + "/" + model_alias + "-" + str(epoch_num))
    net.train()


if "BASE_DIR" not in os.environ:
    os.environ["BASE_DIR"] = "/home/ec2-user/emb3"

random_state = np.random.RandomState(100)

model = ImplicitSequenceModel(loss=h['loss'],
                              representation='mixture',
                              batch_size=h['batch'],
                              learning_rate=h['lr'],
                              l2=h['l2'],
                              n_iter=h['n_iter'],
                              embedding_dim=h['emb_dim'],
                              use_cuda=use_cuda,
                              random_state=random_state,
                              notify_loss_completion=notify_loss_completion,
                              notify_batch_eval_completion=notify_batch_eval_completion,
                              notify_epoch_completion=notify_epoch_completion,
                              log_loss_interval=5000,
                              log_eval_interval=20000,
                              amsgrad=h['amsgrad'],
                              adamw=h['adamw'],
                              betas=betas,
                              num_negative_samples=h['neg'])

logger.info("Model is initialized, now fitting..")
model.fit(train_seq)