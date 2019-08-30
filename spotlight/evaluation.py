import numpy as np

import scipy.stats as st
import pandas as pd
import math
import pickle
import torch
import os

FLOAT_MAX = np.finfo(np.float32).max

suffix = os.environ['SUFFIX']


validate_neg_flatten_vids = pd.read_parquet("data/validate-neg-flatten-aug-28-phase" + suffix)
validate_pos_flatten_vids = pd.read_parquet("data/validate-pos-flatten-aug-28-phase" + suffix)

evaluate_data =  [validate_pos_flatten_vids["uindex"].to_numpy(),
                  validate_pos_flatten_vids["vindex"].to_numpy(),
                  validate_neg_flatten_vids["uindex"].to_numpy(),
                   validate_neg_flatten_vids["nvindex"].to_numpy()]



class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
        # the golden set
        test = pd.DataFrame({'user': test_users,
                             'test_item': test_items,
                             'test_score': test_scores})
        # the full set
        full = pd.DataFrame({'user': neg_users + test_users,
                             'item': neg_items + test_items,
                             'score': neg_scores + test_scores})
        full = pd.merge(full, test, on=['user'], how='left')
        # rank the items according to the scores for each user
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full

    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        return len(test_in_top_k) * 1.0 / full['user'].nunique()

    def cal_ndcg(self):
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(
            lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()

def calc_embs_rank(embeds):
    embeds_norm = np.divide(embeds, np.sqrt(np.square(embeds).sum(axis=1)).reshape(-1, 1))
    cosine_sims = 1 - np.dot(embeds_norm, np.transpose(embeds_norm))
    cosine_sims = cosine_sims.filled(2)
    return pd.DataFrame(cosine_sims).rank(method="first")

def pairs_ndcg_score(embs):
    vindex_pairs_df = pd.read_parquet("data/test-pairs-indexed-aug-28-phase" + suffix)
    embs_ranks = calc_embs_rank(embs)
    number_of_videos = len(embs_ranks)
    lookup_table = embs_ranks.values.ravel()
    ndcg_vals = vindex_pairs_df.apply(
        lambda r: (1. / np.log(lookup_table[r["v1"] * number_of_videos + r["v2"]] + 1.)) * r["count"], axis=1)
    return ndcg_vals.mean()


def calc_als_pairs_ndcg():
    als_embds = pd.read_parquet("s3a://tubi-playground-production/smistry/emb3/als-embs-pandas-aug-28-threshold-0.2")
    aa = pd.read_parquet("data/video2index-pandas-aug-28-phase" + suffix)
    videoid2index = dict(zip(aa["k"], aa["v"]))

    number_of_videos = len(videoid2index)
    embs = np.ma.masked_all((number_of_videos, 100))
    for i, row in als_embds.iterrows():
        vindex = row["vindex"]
        if vindex != -1:
            embs[vindex, :] = row["vector"]
    return pairs_ndcg_score(embs)


def nn_pairs_ndcg_score(model):
    aa = pd.read_parquet("data/video2index-pandas-aug-28-phase" + suffix)
    videoid2index = dict(zip(aa["k"], aa["v"]))

    model.eval()
    number_of_videos = len(videoid2index)
    with torch.no_grad():
        raw_embeds = model.get_embeddings().detach()
        raw_embeds = raw_embeds.cpu()
        raw_embeds = raw_embeds.numpy()
        embed_size = model.get_embedding_size()

        embs = np.ma.masked_all((number_of_videos+10, embed_size))
        for idx, emb in enumerate(raw_embeds):
            #if idx in valid_ids:
            embs[idx, :] = emb
        return pairs_ndcg_score(embs)


def eval_results_in_batch(implicit_model,
                          test_users,
                          test_items,
                          batch_size=1024):

    total_size = len(test_users)
    tmp_ranges = np.arange(0, total_size + batch_size, batch_size)
    lower_indices = tmp_ranges[:-1]
    upper_indices = tmp_ranges[1:]
    subsets = []
    for i in range(len(lower_indices)):
        subset_users = test_users[lower_indices[i]:upper_indices[i]]
        subset_items = test_items[lower_indices[i]:upper_indices[i]]
        if len(subset_users) > 0:
            subsets.append(implicit_model.predict(test_users, test_items))
    return np.concatenate(subsets, 0)

def evaluate_hit_ratio_and_ndcg(implicit_model):
    metron = MetronAtK(top_k=10)
    with torch.no_grad():
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]

        test_scores = eval_results_in_batch(implicit_model, test_users, test_items, batch_size=1024 * 3)
        negative_scores = eval_results_in_batch(implicit_model, negative_users, negative_items, batch_size=1024 * 3)

        metron.subjects = [test_users.tolist(),
                           test_items.tolist(),
                           test_scores.tolist(),
                           negative_users.tolist(),
                           negative_items.tolist(),
                           negative_scores.tolist()]
        hit_ratio, ndcg = metron.cal_hit_ratio(), metron.cal_ndcg()
    return hit_ratio, ndcg


def mrr_score(model, test, train=None):
    """
    Compute mean reciprocal rank (MRR) scores. One score
    is given for every user with interactions in the test
    set, representing the mean reciprocal rank of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, scores of known
        interactions will be set to very low values and so not
        affect the MRR.

    Returns
    -------

    mrr scores: numpy array of shape (num_users,)
        Array of MRR scores for each user in test.
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    mrrs = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            predictions[train[user_id].indices] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[row.indices]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)


def sequence_mrr_score(model, test, exclude_preceding=False):
    """
    Compute mean reciprocal rank (MRR) scores. Each sequence
    in test is split into two parts: the first part, containing
    all but the last elements, is used to predict the last element.

    The reciprocal rank of the last element is returned for each
    sequence.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.SequenceInteractions`
        Test interactions.
    exclude_preceding: boolean, optional
        When true, items already present in the sequence will
        be excluded from evaluation.

    Returns
    -------

    mrr scores: numpy array of shape (num_users,)
        Array of MRR scores for each sequence in test.
    """

    sequences = test.sequences[:, :-1]
    targets = test.sequences[:, -1:]

    mrrs = []

    for i in range(len(sequences)):

        predictions = -model.predict(sequences[i])

        if exclude_preceding:
            predictions[sequences[i]] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[targets[i]]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)


def sequence_precision_recall_score(model, test, k=10, exclude_preceding=False):
    """
    Compute sequence precision and recall scores. Each sequence
    in test is split into two parts: the first part, containing
    all but the last k elements, is used to predict the last k
    elements.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.SequenceInteractions`
        Test interactions.
    exclude_preceding: boolean, optional
        When true, items already present in the sequence will
        be excluded from evaluation.

    Returns
    -------

    mrr scores: numpy array of shape (num_users,)
        Array of MRR scores for each sequence in test.
    """
    sequences = test.sequences[:, :-k]
    targets = test.sequences[:, -k:]
    precision_recalls = []
    for i in range(len(sequences)):
        predictions = -model.predict(sequences[i])
        if exclude_preceding:
            predictions[sequences[i]] = FLOAT_MAX

        predictions = predictions.argsort()[:k]
        precision_recall = _get_precision_recall(predictions, targets[i], k)
        precision_recalls.append(precision_recall)

    precision = np.array(precision_recalls)[:, 0]
    recall = np.array(precision_recalls)[:, 1]
    return precision, recall


def _get_precision_recall(predictions, targets, k):

    predictions = predictions[:k]
    num_hit = len(set(predictions).intersection(set(targets)))

    return float(num_hit) / len(predictions), float(num_hit) / len(targets)


def precision_recall_score(model, test, train=None, k=10):
    """
    Compute Precision@k and Recall@k scores. One score
    is given for every user with interactions in the test
    set, representing the Precision@k and Recall@k of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, scores of known
        interactions will not affect the computed metrics.
    k: int or array of int,
        The maximum number of predicted items
    Returns
    -------

    (Precision@k, Recall@k): numpy array of shape (num_users, len(k))
        A tuple of Precisions@k and Recalls@k for each user in test.
        If k is a scalar, will return a tuple of vectors. If k is an
        array, will return a tuple of arrays, where each row corresponds
        to a user and each column corresponds to a value of k.
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if np.isscalar(k):
        k = np.array([k])

    precision = []
    recall = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            rated = train[user_id].indices
            predictions[rated] = FLOAT_MAX

        predictions = predictions.argsort()

        targets = row.indices

        user_precision, user_recall = zip(*[
            _get_precision_recall(predictions, targets, x)
            for x in k
        ])

        precision.append(user_precision)
        recall.append(user_recall)

    precision = np.array(precision).squeeze()
    recall = np.array(recall).squeeze()

    return precision, recall


def rmse_score(model, test):
    """
    Compute RMSE score for test interactions.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.

    Returns
    -------

    rmse_score: float
        The RMSE score.
    """

    predictions = model.predict(test.user_ids, test.item_ids)

    return np.sqrt(((test.ratings - predictions) ** 2).mean())
