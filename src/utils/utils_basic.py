import logging
import sys
from math import exp
import torch


def getlogger(cfg):
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if cfg["LOGLEVEL"] == "DEBUG":
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG, format=log_fmt, stream=sys.stdout)
        return logger
    elif cfg["LOGLEVEL"] == "INFO":
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
        return logger
    elif cfg["LOGLEVEL"] == "WARNING":
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.WARNING, format=log_fmt, stream=sys.stdout)
        return logger
    elif cfg["LOGLEVEL"] == "ERROR":
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.ERROR, format=log_fmt, stream=sys.stdout)
        return logger
    else:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
        logger.info(
            f'loglevel: {cfg["LOGLEVEL"]} must be DEBUG, INFO, WARNING OR ERROR, setting to INFO'
        )
        return logger


def read_ont_file(file_path, sep="\t"):
    """Function to read-in text files of ontologies with format child - separator - parent into an dictionary.
    ARGS:
        file_path - (str): Path to file with ontology
        sep - (str): Separator used in file
    RETURNS:
        ont_dic - (dict): Dictionary containing the ontology as described in the text file.

    """
    ont_dic = dict()
    with open(file_path, "r") as ont_file:
        for line in ont_file:
            id_parent = line.strip().split(sep)[1]
            id_child = line.split(sep)[0]

            if id_parent in ont_dic:
                ont_dic[id_parent].append(id_child)
            else:
                ont_dic[id_parent] = list()
                ont_dic[id_parent].append(id_child)

    return ont_dic


def get_device(verbose=True):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        torch.use_deterministic_algorithms(False)
    return device


def annealer(epoch_current, total_epoch, func="logistic-mid"):
    """Defines VAE loss annealing function
    ARGS:
        epoch_current - (int): current epoch in training
        total_epoch - (int): total epochs for training
        func - (str): specification of annealing function.
        Default is "logistic-mid" with logistic increase of annealing and midpoint at half of total epochs.
    RETURNS:
        anneal_weight - (float): Annealing weight. Between 0 (no VAE loss) and 1 (full VAE loss).

    """

    if epoch_current == None:
        anneal_weight = 1
    else:
        match func:
            case "3phase-linear":
                if epoch_current < total_epoch / 3:
                    anneal_weight = 0

                else:
                    if epoch_current < 2 * (total_epoch / 3):
                        anneal_weight = (epoch_current - total_epoch / 3) / (
                            total_epoch / 3
                        )
                    else:
                        anneal_weight = 1
            case "3phase-log":
                if epoch_current < total_epoch / 3:
                    anneal_weight = 0

                else:
                    if epoch_current < 2 * (total_epoch / 3):
                        anneal_weight = annealer(
                            epoch_current=epoch_current - (total_epoch / 3),
                            total_epoch=(total_epoch / 3),
                            func="logistic-mid",
                        )
                    else:
                        anneal_weight = 1
            case "logistic-mid":
                B = (1 / total_epoch) * 20
                M = 0.5
                anneal_weight = 1 / (1 + exp(-B * (epoch_current - total_epoch * M)))
            case "logistic-early":
                B = (1 / total_epoch) * 20
                M = 0.25
                anneal_weight = 1 / (1 + exp(-B * (epoch_current - total_epoch * M)))
            case "logistic-late":
                B = (1 / total_epoch) * 20
                M = 0.75
                anneal_weight = 1 / (1 + exp(-B * (epoch_current - total_epoch * M)))
            case "no-annealing":
                anneal_weight = 1
            case _:
                raise NotImplementedError("The annealer is not implemented yet")

    return anneal_weight


def get_annealing_epoch(cfg, current_epoch):
    """checks if annealing should be used for pretraining
    ARGS:
        cfg - (dict): configuration dictionary
        current_epoch - (int): current epoch
    RETURNS:
        int: annealing epoch

    """
    if not cfg["ANNEAL_PRETRAINING"]:
        if current_epoch <= cfg["PRETRAIN_EPOCHS"]:
            return None
        else:
            # to discard pretraining epochs from beta_annealing process
            return current_epoch - cfg["PRETRAIN_EPOCHS"]
    else:
        if current_epoch <= cfg["PRETRAIN_EPOCHS"]:
            return current_epoch
        else:
            return current_epoch - cfg["PRETRAIN_EPOCHS"]
