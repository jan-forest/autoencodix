import yaml
import os
from src.utils.utils_basic import getlogger

def get_cfg(run_id):
    """A function to read YAML file
    Args:
        rund_id (str): ID of the run
    Returns:
        config (dict): a dictionary of configuration parameters

    """
    
    with open(os.path.join("reports", f"{run_id}", f"{run_id}_config.yaml")) as f:
        config = yaml.safe_load(f)
    config["RUN_ID"] = run_id

    with open(os.path.join("src", "000_internal_config.yaml")) as f:
        config_internal = yaml.safe_load(f)

    # config.update(config_internal)
    # return config
    config_internal.update(
        config
    )  ### Switch order to be able to overwrite internal config params with normal config
    logger = getlogger(cfg=config_internal)
    if config_internal["BATCH_SIZE"] == 1:
        logger.warning("BATCH_SIZE=1 is not compatible to batch normalization layer. BATCH_SIZE is set to 2.")
        config_internal["BATCH_SIZE"] = 2
    return config_internal
