# @Time   : 2022/3/12
# @Author : zihan Lin
# @Email  : zhlin@ruc.edu.cn

"""
recbole_cdr.quick_start
########################
"""
import logging
from logging import getLogger
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

from recbole.utils import init_logger, init_seed, set_color
from recbole_cdr.config import CDRConfig
from recbole.data.interaction import Interaction
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.utils import get_model, get_trainer


def run_recbole_cdr(model=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = CDRConfig(model=model, config_file_list=config_file_list, config_dict=config_dict)

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    total_target_user = len(list(test_data.uid_list.numpy()))
    logger.info(set_color('target user num ', 'yellow') + f': {total_target_user}')

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # trainer.resume_checkpoint('/home/hadh2/projects/cross_domain/RecBole-CDR/saved/SSCDR-May-08-2024_12-01-10.pth')
    # return trainer.config
    # user_id_col_name = trainer.model.TARGET_USER_ID
    # item_id_col_name = trainer.model.TARGET_ITEM_ID
    # num_test_users = len(list(test_data.uid_list.numpy()))
    # user_item = list(test_data.uid2positive_item)
    # user_item.pop(0)
    # num_test_items = max(set([item for items in user_item for item in list(items.numpy())]))

    # results = {}

    # for user_id in tqdm(list(range(1, num_test_users + 1)), desc='Predicting users'):
    #     users = [user_id]
    #     items = list(range(1, num_test_items + 1))

    #     interactions = [(user, item) for user in users for item in items]
    #     interaction_df = pd.DataFrame(interactions, columns=[user_id_col_name, item_id_col_name])
    #     interaction = Interaction(interaction_df)

    #     output = trainer.model.predict(interaction)
    #     output = list(output.detach().cpu().numpy())

    #     interaction_df = pd.DataFrame(interaction.numpy())
    #     output_df = pd.DataFrame({'score': output})
    #     df = pd.concat([interaction_df, output_df], axis=1)

    #     top_10_items = df.nlargest(10, 'score')[item_id_col_name].to_list()
    #     real_items = list(test_data.uid2positive_item[user_id].numpy())

    #     results[user_id] = {
    #         'predict': top_10_items,
    #         'true': real_items,
    #         'hit': list(set(top_10_items) & set(real_items))
    #     }

    # return results
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('target user num ', 'yellow') + f': {total_target_user}')
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = CDRConfig(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data
