import os
import logging
import numpy as np
import random
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_logger(save=True, log_path=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if save and not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    if not logger.handlers:
        if save:
            # Logging to file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(message)s'))
            logger.addHandler(file_handler)
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    return logger


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=True, delta=0, logger=None):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_f1_max = float('-inf')
        self.delta = delta
        self.logger = logger

    def __call__(self, f1_score, model):

        if self.best_score is None:
            self.best_score = f1_score
            self.save_checkpoint(f1_score, model)
        elif f1_score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = f1_score
            self.save_checkpoint(f1_score, model)
            self.counter = 0

    def save_checkpoint(self, f1_score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.logger.info(f'Validation f1 increased ({self.val_f1_max:.4f} --> {f1_score:.4f}).  Saving model ...')
        path = os.path.join(self.save_path, 'model.pth')
        torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        self.val_f1_max = f1_score


