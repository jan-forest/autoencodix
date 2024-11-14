import os
import pathlib

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.make_dataset import normalize_data
from src.utils.utils_basic import get_device, getlogger


class ImageScaler:
    """Finds nearest quadratic image size that is dividable by 2^number_of_layers
    to the given image size in the cfg file

    """

    def __init__(self, cfg):
        self.cfg = cfg
        found_image_type = False
        for data_type in cfg["DATA_TYPE"]:
            if cfg["DATA_TYPE"][data_type]["TYPE"] == "IMG":
                self.width = cfg["DATA_TYPE"][data_type]["WIDTH"]
                self.height = cfg["DATA_TYPE"][data_type]["HEIGHT"]
                found_image_type = True
        if not found_image_type:
            raise ValueError("You need to provide a DATA_TYPE of with the TYPE key IMG")

        if not self.width == self.height:
            raise ValueError("Image width and height need to be the same")
        # check for int type in width and height
        if not isinstance(self.width, int):
            raise ValueError("Image width and height need to be integers")
        if not isinstance(self.width, int):
            raise ValueError("Image width and height need to be integers")
        self.tuning = self.cfg["TRAIN_TYPE"] == "tune"
        self.n_conv_layers = 5
        if self.tuning:
            self.n_conv_layers = self.cfg["LAYERS_UPPER_LIMIT"]
        self.logger = getlogger(cfg=self.cfg)

    def _image_size_is_legal(self):
        """Checks if image size is dividable by 2^number_of_layers
        to the given image size in the cfg file

        """
        return self.width % 2**self.n_conv_layers == 0

    def get_nearest_quadratic_image_size(self):
        """Finds nearest quadratic image size that is dividable by 2^number_of_layers
        to the given image size in the cfg file

        """
        if self._image_size_is_legal():
            self.logger.info(
                f"Given image size is possible, rescaling images to: {self.width}x{self.height}"
            )
            return self.width, self.height
        running_image_size = self.width
        while_loop_counter = 0
        while not running_image_size % 2**self.n_conv_layers == 0:
            running_image_size += 1
            while_loop_counter += 1
            if while_loop_counter > 10000:
                raise ValueError(
                    f"Could not find a quadratic image size that is dividable by 2^{self.n_conv_layers}"
                )
        self.logger.info(
            f"Given image size{self.width}x{self.height} is not possible, rescaling to: {running_image_size}x{running_image_size}"
        )
        return running_image_size, running_image_size


class DataSetPreparer:
    def __init__(self, cfg, data_modality, split_type, path_key=""):
        self.cfg = cfg
        self.data_modality = data_modality
        self.split_type = split_type
        self.path_key = path_key
        self.X = self.read_data_file()
        splitter = FilterBasedOnSplit(cfg=cfg, split_type=self.split_type, X=self.X)
        self.X_split = splitter.get_data()
        self.sample_ids = self.X_split.index.tolist()
        if self.data_modality == "IMG":
            image_scaler = ImageScaler(self.cfg)
            (
                self.scale_width,
                self.scale_height,
            ) = image_scaler.get_nearest_quadratic_image_size()
            self.X_tensor = self.image_data_to_tensor()

        elif self.data_modality in ["NUMERIC", "MIXED", "COMBINED", "CONCAT"]:
            self.X_tensor = self.numeric_data_to_tensor()
        else:
            raise ValueError(
                f"Data modality {self.data_modality} not supported, use one of ['IMG', 'NUMERIC', 'MIXED', 'COMBINED', 'CONCAT']"
            )

    def get_data(self):
        return self.sample_ids, self.X_tensor, self.X_split

    def read_data_file(self):
        
        file_path = os.path.join(
            "data/processed",
            self.cfg["RUN_ID"],
            self.path_key + "_data.parquet",
        )

        # Interim data set file with combined input
        if self.path_key.split("-")[0] == "COMBINED":
            file_path = os.path.join(self.cfg["DATA_TYPE"][self.path_key]["FILE_PROC"])

        if self.path_key.split("-")[0] == "CONCAT":            
            # we need this, because the HVAE Input are the concated latent spaces of the VAEs for each datatype

            file_path = os.path.join(self.cfg["DATA_TYPE"][self.path_key]["FILE_PROC"])
            X = pd.read_parquet(file_path)
            if not self.cfg["RECONSTR_LOSS"] == "MSE":   # # using min max Scalinf for HVAE input, when BCE or other loss.
                X = normalize_data(X, cfg=self.cfg, method="MinMax")

            return X

        return pd.read_parquet(file_path)

    def numeric_data_to_tensor(self):
        X_interim = np.vstack(self.X_split.values).astype(float)
        device = get_device()
        if torch.cuda.is_available():
            torch.use_deterministic_algorithms(False)
        return torch.tensor(X_interim, dtype=torch.float32).to(device)

    def get_scaled_dimensions(self):
        if not self.data_modality == "IMG":
            raise NotImplementedError(
                "This function is only implemented for image data"
            )
        else:
            return self.scale_width, self.scale_height

    def image_data_to_tensor(self):
        images = []
        image_paths = self.X_split["img_paths"].tolist()

        for img_path in image_paths:
            image_path = os.path.join(self.cfg["ROOT_IMAGE"], img_path).replace(
                "\t", ""
            )
            images.append(self.parse_image_to_tensor(image_path))
        return images

    def parse_image_to_tensor(self, image_path):
        supported_extensions = [
            "jpg",
            "jpeg",
            "JPEG",
            "JPG",
            "png",
            "PNG",
            "tif",
            "TIF",
            "tiff",
            "TIFF",
        ]

        # Extract file extension from the image path
        file_extension = image_path.split(".")[-1].lower()
        if file_extension not in supported_extensions:
            raise ValueError(
                f"Unsupported image format: {file_extension}. Enabled are: \n {supported_extensions}"
            )

        # Handle special cases for certain image formats
        if file_extension in ["tif", "tiff"]:
            # Read TIFF images with cv2.IMREAD_UNCHANGED
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        else:
            # Read other image formats normally
            image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Resize the image to in config specified size
        if not len(image.shape) in [2, 3]:
            raise ValueError(
                f"Image has unsupported shape: {image.shape}, supported are 2D and 3D images"
            )
        image = cv2.resize(image, (self.scale_width, self.scale_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert BGR to RGB format
        # check if format is (C, H, W)
        if len(image.shape) == 2:
            # add channel dimension
            image = np.expand_dims(image, axis=2)
        if not image.shape[0] == 3:
            image = image.transpose(2, 0, 1)
        # normalize image
        image = cv2.normalize(
            image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
        image = np.clip(image, 0, 1)
        outpath = os.path.join(
            "data/processed", self.cfg["RUN_ID"], os.path.basename(image_path)
        )
        if not os.path.exists(outpath):
            image_out = image.transpose(1, 2, 0)
            cv2.imwrite(outpath, image_out)
        return torch.from_numpy(image.astype(np.float32))


class UnlabelledNumericDataset(Dataset):
    def __init__(
        self,
        cfg,
        split_type,
        data_modality="NUMERIC",
        path_key="none-none",
    ):
        self.cfg = cfg
        self.split_type = split_type
        if torch.cuda.is_available():
            torch.use_deterministic_algorithms(False)
        dataset_preparer = DataSetPreparer(
            cfg=self.cfg,
            data_modality=data_modality,
            split_type=self.split_type,
            path_key=path_key,
        )

        self.sample_ids, self.x_tensor, self.x_pandas = dataset_preparer.get_data()
        self.data = self.x_pandas  # to stay consistent with previous implementation

        self._input_size = self.x_tensor.shape[1]
        self.data_modality = data_modality
        self.path_key = path_key

    def __getitem__(self, index):
        return self.x_tensor[index], self.sample_ids[index]

    def __len__(self):
        return len(self.x_tensor)

    def get_total_number_of_features(self):
        """Retruns number of features of dataset
        ARGS:
            None
        RETURNS:
            total_number_of_features (int)

        """
        return self.x_tensor.shape[0] * self.x_tensor.shape[1]

    def shape(self):
        return self.x_tensor.shape

    def data_modality(self):
        return self.data_modality

    def get_sample_ids(self):
        return self.sample_ids

    def get_cols(self):
        return self.x_pandas.columns

    def input_size(self):
        return self._input_size


class ImageDataset(Dataset):
    """
    A class for loading imaga data

    ...
    Attributes
    ----------
    cfg : dict
        configuration dictionary
    data_modality : str
        data modality of the dataset
    transform : callable
        transformation function to apply to the data

    Methods
    -------
    __getitem__(index)
        returns the data and the sample id at the given index
    __len__()
        returns the length of the dataset
    get_sample_ids()
        returns the sample ids of the dataset
    get_cols()
        returns the column names of the dataset
    parse_image_to_tensor(image_path)
        parses an image to a tensor
    """

    def __init__(
        self,
        cfg,
        data_modality="IMG",
        transform=None,
        split_type="train",
        path_key="none-none",
    ):
        super().__init__()
        self.cfg = cfg
        self.data_modality = data_modality
        self.split_type = split_type
        datset_preparer = DataSetPreparer(
            cfg=cfg,
            data_modality=data_modality,
            split_type=self.split_type,
            path_key=path_key,
        )
        self.width, self.height = datset_preparer.get_scaled_dimensions()
        self.sample_ids, self.images, self.sample_mappings = datset_preparer.get_data()

    def __getitem__(self, index):
        try:
            device = get_device()
            return self.images[index].to(device), self.sample_ids[index]
        except Exception as e:
            print(e)
            print(f"index: {index}")
            print(f"len(images): {len(self.images)}")
            print(f"ids: {len(self.sample_ids)}")

    def __len__(self):
        return len(self.images)

    def get_total_number_of_features(self):
        """Retruns number of features or mulitplied image dimensions (each pixel is a feature)
        of dataset for IMAGE case
        ARGS:
            None
        RETURNS:
            total_number_of_features (int)

        """
        (img_dimension) = self.input_size()
        # multiply alll dimensions
        total_number_of_features = 1
        for dim in img_dimension:
            total_number_of_features *= dim
        total_number_of_features = total_number_of_features * len(self.sample_ids)

    def get_sample_ids(self):
        return self.sample_ids

    def input_size(self):
        return self.images[0].shape

    def data_modality(self):
        return self.data_modality

    def get_cols(self):
        cols = [f"pixel_{i}" for i in range(self.width)]
        return cols


class FilterBasedOnSplit:
    def __init__(self, cfg, split_type, X):
        self.cfg = cfg
        self.split_type = split_type
        self.filtered_index = self.get_index_split()
        try:
            self.filtered_X = X.loc[self.filtered_index]
        except KeyError:
            logger = getlogger(cfg=self.cfg)
            logger.error(
                f"There are sample ID's from split type {split_type} which are not in the input data."
            )
            logger.error(f"Check your SPLIT_FILE if you selected a pre-computed split")
            raise

    def get_index_split(self):
        if pathlib.Path(self.cfg["SPLIT_FILE"]).suffix in [".csv", ".tsv", ".txt"]:
            if self.cfg["SPLIT"] == "pre-computed":
                sample = pd.read_csv(
                    self.cfg["SPLIT_FILE"],
                    index_col=0,
                    sep=self.cfg["DELIM"],
                )
            else:
                sample = pd.read_csv(
                    os.path.join(
                        os.path.dirname(self.cfg["SPLIT_FILE"]),
                        self.cfg["RUN_ID"],
                        os.path.basename(self.cfg["SPLIT_FILE"]),
                    ),
                    index_col=0,
                    sep=self.cfg["DELIM"],
                )

        elif pathlib.Path(self.cfg["SPLIT_FILE"]).suffix == ".parquet":
            if self.cfg["SPLIT"] == "pre-computed":
                sample = pd.read_parquet(
                    self.cfg["SPLIT_FILE"],
                )
            else:
                sample = pd.read_parquet(
                    os.path.join(
                        os.path.dirname(self.cfg["SPLIT_FILE"]),
                        self.cfg["RUN_ID"],
                        os.path.basename(self.cfg["SPLIT_FILE"]),
                    )
                )
        else:
            raise ValueError(
                f"You provided a not supported input file type:{pathlib.Path(self.cfg['SPLIT_FILE']).suffix}"
            )
        # for predict case
        if self.split_type == "all":
            return sample.index
        if self.split_type in sample["SPLIT"].unique():
            index_split = sample[sample["SPLIT"] == self.split_type].index
            return index_split
        else:
            logger = getlogger(cfg=self.cfg)
            logger.warning(
                f"You provided a not supported split type:{self.split_type} or the split type is not in the split file. train, valid, test are supported"
            )
            logger.warning(f"Preparing Dataloader with all data")
            index_split = sample.index
            return index_split

    def get_data(self):
        return self.filtered_X


class CrossModaleDataset(Dataset):
    def __init__(self, cfg, split_type="train"):
        self.cfg = cfg
        self.split_type = split_type

        self.from_key, self.to_key = cfg["TRANSLATE"].split("_to_")
        if self.from_key not in cfg["DATA_TYPE"].keys():
            raise ValueError(
                f"TRANSLATE config parameter {self.from_key} needs\
                to be consistent with DATA_TYPE config parameter chose on of\
                    {self.cfg['DATA_TYPE'].keys()}"
            )
        if self.to_key not in cfg["DATA_TYPE"].keys():
            raise ValueError(
                f"TRANSLATE config parameter {self.to_key} needs\
                to be consistent with DATA_TYPE config parameter chose on of\
                    {self.cfg['DATA_TYPE'].keys()}"
            )
        self.from_modality = cfg["DATA_TYPE"][self.from_key]["TYPE"]
        self.to_modality = cfg["DATA_TYPE"][self.to_key]["TYPE"]

        self.translate_from_image = "IMG" == self.from_modality
        self.translate_to_image = "IMG" == self.to_modality

        self.from_data_preparer = DataSetPreparer(
            cfg=cfg,
            data_modality=self.from_modality,
            split_type=self.split_type,
            path_key=self.from_key,
        )
        (
            self.from_sample_ids,
            self.from_tensors,
            self.from_pandas,
        ) = self.from_data_preparer.get_data()

        self.to_data_preparer = DataSetPreparer(
            cfg=cfg,
            data_modality=self.to_modality,
            split_type=self.split_type,
            path_key=self.to_key,
        )
        (
            self.to_sample_ids,
            self.to_tensors,
            self.to_pandas,
        ) = self.to_data_preparer.get_data()
        (
            self.paired_sample_ids,
            self.paired_from_tensors,
            self.paired_to_tensors,
        ) = self.pair_sample_ids()
        self.from_loss_scaler, self.to_loss_scaler = self.get_loss_scaler()

    def pair_sample_ids(self):
        """Takes a list of sample_ids from a to_data modality and a from_data
        modality, and corresponding tensors, and makes sure all are in the same order.
        """
        from_id_to_index = {
            sample_id: idx for idx, sample_id in enumerate(self.from_sample_ids)
        }
        paired_from_tensors = []
        paired_to_tensors = []
        paired_sample_ids = []
        for to_sample_id in self.to_sample_ids:
            if to_sample_id in from_id_to_index:
                idx = from_id_to_index[to_sample_id]
                paired_sample_ids.append(to_sample_id)
                paired_from_tensors.append(self.from_tensors[idx])
                paired_to_tensors.append(
                    self.to_tensors[self.to_sample_ids.index(to_sample_id)]
                )
        return paired_sample_ids, paired_from_tensors, paired_to_tensors

    def __len__(self):
        return len(self.paired_sample_ids)

    def __getitem__(self, idx):
        return (
            self.paired_sample_ids[idx],
            self.paired_from_tensors[idx],
            self.paired_to_tensors[idx],
        )

    def get_loss_scaler(self):
        if not self.cfg["NORMALIZE_RECONS_LOSS_XMODALIX"]:
            return 0.5, 0.5
        to_number_of_features = self.get_total_number_of_features("to")
        from_number_of_features = self.get_total_number_of_features("from")
        from_loss_scaler = (to_number_of_features) / (
            from_number_of_features + to_number_of_features
        )
        to_loss_scaler = (from_number_of_features) / (
            from_number_of_features + to_number_of_features
        )
        return from_loss_scaler, to_loss_scaler

    def get_total_number_of_features(self, direction):
        """Retruns number of features or mulitplied image dimensions (each pixel is a feature)
        of dataset for IMAGE case
        ARGS:
            direction - (str): "to", "from"
        RETURNS:
            total_number_of_features (int)

        """
        if direction == "from":
            if self.from_modality == "IMG":
                (img_dimension) = self.from_tensors[0].shape
                total_number_of_features = 1
                for dim in img_dimension:
                    total_number_of_features *= dim
                return total_number_of_features
            return self.from_tensors.shape[1]
        elif direction == "to":
            if self.to_modality == "IMG":
                img_dimension = self.to_tensors[0].shape
                total_number_of_features = 1
                for dim in img_dimension:
                    total_number_of_features *= dim
                return total_number_of_features
            return self.to_tensors.shape[1]
        else:
            raise ValueError(f"direction  {direction} needs to be 'to' or 'from'")

    def input_size(self, direction):
        """Retruns number of features or image dimension of dataset for IMAGE case
         ARGS:
             direction - (str): "to", "from"
        RETURNS:
             input_size (tupe or int)
        """
        if direction == "from":
            if self.from_modality == "IMG":
                return self.from_tensors[0].shape
            else:
                return self.from_tensors.shape[1]
        elif direction == "to":
            if self.to_modality == "IMG":
                return self.to_tensors[0].shape
            else:
                return self.to_tensors.shape[1]

        else:
            raise ValueError(f"direction  {direction} needs to be 'to' or 'from'")

    def get_cols(self, direction):
        if direction == "from":
            if self.from_modality == "IMG":
                width = self.from_data_preparer.get_scaled_dimensions()[0]
                return [f"pixel_{i}" for i in range(width)]
            else:
                self.from_pandas.columns
        elif direction == "to":
            if self.to_modality == "IMG":
                width = self.to_data_preparer.get_scaled_dimensions()[0]
                return [f"pixel_{i}" for i in range(width)]
            else:
                return self.to_pandas.columns

    def get_sample_ids(self):
        return self.pair_sample_ids

    def get_to_cols(self):
        pass

    def get_from_cols(self):
        pass
