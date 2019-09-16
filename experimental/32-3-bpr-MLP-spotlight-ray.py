import ray
from ray import tune
import os
import random
os.environ['SUFFIX']="500K-with-3-vids"
os.environ['BASE_DIR']="/home/ec2-user/emb3"

from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.interactions import Interactions
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.tensorboard import SummaryWriter
from spotlight.factorization.representations import *
import os
from spotlight.evaluation import *
from spotlight.evaluation import mrr_score

def ray_train(input_config, reporter):
    if 'SUFFIX' in os.environ:
        suffix = os.environ['SUFFIX']
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

    betas=(mom, 0.999)
    use_cuda=True

    tensorboard_base_dir="/home/ec2-user/emb3/runs"
    model_alias = "{}-suf-{}-loss-{}-lr-{}-l2-{}-mom-{}-smpl-{}-ng-{}-bt-{}-drp-{}-ray".format(net_conf, suffix, loss, lr, l2, 
                                                                                          mom, train_sample,
                                                                                          num_negative_samples,
                                                                                         batch_size,
                                                                                                        dropout)
    model_store_dir="/home/ec2-user/emb3/models"
    n_iters=30
    #loss="adaptive_hinge"

    log_loss_interval=100
    log_eval_interval=5000
    #train_data_path = "s3a://tubi-playground-production/smistry/emb3/train-aug-28-phase1"
    train_data_path = os.environ['BASE_DIR'] + "/data/train-aug-28-phase" + suffix

    
    original_train_data = pd.read_parquet(train_data_path)
    writer = SummaryWriter(log_dir='{}/{}'.format(tensorboard_base_dir, model_alias))
    writer.add_text('alias', model_alias, 0)

    def notify_loss_completion(epoch_id, batch_id, loss, net, model):
        #print("notify_loss_completion")
        writer.add_scalar("Batch/loss", loss, batch_id)

    def notify_batch_eval_completion(epoch_id, batch_id, loss, net, model):
        #print("notify_batch_eval_completion")
        pairs_ndcg = nn_pairs_ndcg_score(net)
        reporter(mean_accuracy=pairs_ndcg, timesteps_total=int(batch_id/log_eval_interval) * 10, checkpoint=model_alias)
        writer.add_scalar("Batch/pairs_ndcg", pairs_ndcg, batch_id)


    def notify_epoch_completion(epoch_num, total_loss, net, model):
        #print("notify_epoch_completion")
        writer.add_scalar("Epoch/loss", total_loss, epoch_num)
        pairs_ndcg = nn_pairs_ndcg_score(net)
        writer.add_scalar("Epoch/pairs_ndcg", pairs_ndcg, epoch_num)
    #     hit_ratio, ndcg = evaluate_hit_ratio_and_ndcg(model)
    #     writer.add_scalar("Epoch/HR", hit_ratio, epoch_num)
    #     writer.add_scalar("Epoch/NDCG", ndcg, epoch_num)
        hit_ratio, ndcg = -1,-1
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
                                      representation=rep)
    model.fit(interactions)
    
    
NUM_GPUS=8
ray.init(ignore_reinit_error=True, num_gpus=NUM_GPUS)

from ray.tune.schedulers import AsyncHyperBandScheduler


#exp_config = {
#    "LOSS": tune.grid_search(["adaptive_hinge"]),
#    "LR": tune.grid_search([1e-3, 1e-4, 1e-2, 0.005]),
#    "L2": tune.grid_search([1e-5, 1e-8, 1e-10]),
#    "DROPOUT": tune.grid_search([0.1,0.2,0.4,0.5]),
#    "MOM": tune.grid_search([0.9,0.92,0.95,0.8]),
#    "NEGSAMPLES": tune.grid_search([5, 3, 10])
#}

exp_config = {
    "LOSS": tune.sample_from(lambda _: random.choice(["adaptive_hinge"])),
    "LR": tune.sample_from(lambda _: random.choice([1e-3, 1e-4, 1e-2, 0.005])),
    "L2": tune.sample_from(lambda _: random.choice([1e-5, 1e-8, 1e-10])),
    "DROPOUT": tune.sample_from(lambda _: random.choice([0.1,0.2,0.4,0.5])),
    "MOM": tune.sample_from(lambda _: random.choice([0.9,0.92,0.95,0.8])),
    "NEGSAMPLES": tune.sample_from(lambda _: random.choice([5, 3, 8, 10]))
}

configuration = tune.Experiment(
    "check_for_500K_3_vids",
    run=ray_train,
    num_samples=1,
    resources_per_trial={"gpu": 1},
    stop={"mean_accuracy": 0.95},  # TODO: Part 1
    config=exp_config
)

import sys
hyperband = AsyncHyperBandScheduler(
    time_attr='timesteps_total',
    reward_attr='mean_accuracy')
trials = tune.run_experiments(configuration, scheduler=hyperband, verbose=True, reuse_actors=True)

def get_sorted_trials(trial_list, metric):
    return sorted(trial_list, key=lambda trial: trial.last_result.get(metric, 0), reverse=True)
  
sorted_trials = get_sorted_trials(trials, metric="mean_accuracy")
print(str([(x.last_result.get("mean_accuracy", 0),  x.last_result.get("iterations_since_restore"), x) for x in sorted_trials]))