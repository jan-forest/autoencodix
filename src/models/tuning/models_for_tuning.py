import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.models import FCLayer_1D, get_layer_dim


class VarixTune(nn.Module):
    def __init__(self, trial, input_dim, cfg, latent_dim=None):
        super(VarixTune, self).__init__()
        if latent_dim == None:
            latent_dim = cfg["LATENT_DIM_FIXED"]
        self.latent_dim = latent_dim
        self.trial = trial
        self.input_dim = input_dim
        self.cfg = cfg
        self.n_layers = self.trial.suggest_int(
            "fc_n_layers",
            max([cfg["LAYERS_LOWER_LIMIT"], 2]),
            cfg["LAYERS_UPPER_LIMIT"],
        )

        self.global_p = self.trial.suggest_float(
            f"dropout_all", cfg["DROPOUT_LOWER_LIMIT"], cfg["DROPOUT_UPPER_LIMIT"]
        )

        self.enc_factor = self.trial.suggest_float(
            f"encoding_factor",
            cfg["ENCODING_FACTOR_LOWER_LIMIT"],
            cfg["ENCODING_FACTOR_UPPER_LIMIT"],
        )

        # Building architecture
        enc_dim = get_layer_dim(
            self.input_dim, self.latent_dim, self.n_layers, self.enc_factor
        )
        encoder_layers = []
        [
            encoder_layers.extend(FCLayer_1D(enc_dim[i], enc_dim[i + 1], self.global_p))
            for i in range(len(enc_dim) - 2)
        ]

        dec_dim = enc_dim[::-1]
        decoder_layers = []
        [
            decoder_layers.extend(FCLayer_1D(dec_dim[i], dec_dim[i + 1], self.global_p))
            for i in range(len(dec_dim) - 2)
        ]
        decoder_layers.extend(
            FCLayer_1D(dec_dim[-2], dec_dim[-1], self.global_p, only_linear=True)
        )  ## No ReLU and dropout in last decoder layer

        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(enc_dim[-2], self.latent_dim)
        self.logvar = nn.Linear(enc_dim[-2], self.latent_dim)
        self.decoder = nn.Sequential(*decoder_layers)

        # Initialisierung
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def encode(self, x):
        # latent = F.relu(self.encoder(x)) # ReLU before mu and logvar layer
        latent = self.encoder(x)  # ReLU already in encoder definition

        mu = self.mu(latent)
        logvar = self.logvar(latent)
        # prevent  mu and logvar from beeing to close to zero this increased
        # numerical stability
        logvar = torch.clamp(logvar, 0.1, 20)
        # replace mu when mu < 0.00000001 with 0.1
        mu = torch.where(mu < 0.000001, torch.zeros_like(mu), mu)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VanillixTune(nn.Module):
    def __init__(self, trial, input_dim, cfg, latent_dim=None):
        super(VanillixTune, self).__init__()
        if latent_dim == None:
            latent_dim = cfg["LATENT_DIM_FIXED"]
        self.latent_dim = latent_dim
        self.trial = trial
        self.input_dim = input_dim
        self.cfg = cfg
        self.n_layers = self.trial.suggest_int(
            "fc_n_layers",
            max([cfg["LAYERS_LOWER_LIMIT"], 2]),
            cfg["LAYERS_UPPER_LIMIT"],
        )

        self.global_p = self.trial.suggest_float(
            f"dropout_all", cfg["DROPOUT_LOWER_LIMIT"], cfg["DROPOUT_UPPER_LIMIT"]
        )

        lower = 1
        upper = 64
        self.enc_factor = self.trial.suggest_float(f"encoding_factor", lower, upper)

        # Building architecture
        enc_dim = get_layer_dim(
            self.input_dim, self.latent_dim, self.n_layers, self.enc_factor
        )
        encoder_layers = []
        [
            encoder_layers.extend(FCLayer_1D(enc_dim[i], enc_dim[i + 1], self.global_p))
            for i in range(len(enc_dim) - 2)
        ]
        encoder_layers.extend(
            FCLayer_1D(enc_dim[-2], enc_dim[-1], self.global_p, only_linear=True)
        )  ## No ReLU and dropout in last encoder layer

        dec_dim = enc_dim[::-1]
        decoder_layers = []
        [
            decoder_layers.extend(FCLayer_1D(dec_dim[i], dec_dim[i + 1], self.global_p))
            for i in range(len(dec_dim) - 2)
        ]
        decoder_layers.extend(
            FCLayer_1D(dec_dim[-2], dec_dim[-1], self.global_p, only_linear=True)
        )  ## No ReLU and dropout in last decoder layer

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        # Initialisierung
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def encode(self, x):
        return self.encoder(x), None

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        latent, _ = self.encode(x)
        recon = self.decode(latent)
        return recon, torch.tensor(0), torch.tensor(0)  # _ to be consistent with VAE


class StackixTune(nn.Module):
    def __init__(self, trial, input_dim, cfg, latent_dim=None):
        super(StackixTune, self).__init__()
        if latent_dim == None:
            latent_dim = cfg["LATENT_DIM_FIXED"]
        self.latent_dim = latent_dim
        self.trial = trial
        self.input_dim = input_dim
        self.cfg = cfg
        self.n_layers = self.trial.suggest_int(
            "fc_n_layers",
            max([cfg["LAYERS_LOWER_LIMIT"], 2]),
            cfg["LAYERS_UPPER_LIMIT"],
        )

        self.global_p = self.trial.suggest_float(
            f"dropout_all", cfg["DROPOUT_LOWER_LIMIT"], cfg["DROPOUT_UPPER_LIMIT"]
        )

        lower = 1
        upper = 64
        self.enc_factor = self.trial.suggest_float(f"encoding_factor", lower, upper)

        # Building architecture
        enc_dim = get_layer_dim(
            self.input_dim, self.latent_dim, self.n_layers, self.enc_factor
        )
        encoder_layers = []
        [
            encoder_layers.extend(FCLayer_1D(enc_dim[i], enc_dim[i + 1], self.global_p))
            for i in range(len(enc_dim) - 2)
        ]

        dec_dim = enc_dim[::-1]
        decoder_layers = []
        [
            decoder_layers.extend(FCLayer_1D(dec_dim[i], dec_dim[i + 1], self.global_p))
            for i in range(len(dec_dim) - 2)
        ]
        decoder_layers.extend(
            FCLayer_1D(dec_dim[-2], dec_dim[-1], self.global_p, only_linear=True)
        )  ## No ReLU and dropout in last decoder layer

        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(enc_dim[-2], self.latent_dim)
        self.logvar = nn.Linear(enc_dim[-2], self.latent_dim)
        self.decoder = nn.Sequential(*decoder_layers)

        # Initialisierung
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def encode(self, x):
        # latent = F.relu(self.encoder(x)) # ReLU before mu and logvar layer
        latent = self.encoder(x)  # ReLU already in encoder definition

        mu = self.mu(latent)
        logvar = self.logvar(latent)
        # prevent  mu and logvar from beeing to close to zero this increased
        # numerical stability
        logvar = torch.clamp(logvar, 0.1, 20)
        # replace mu when mu < 0.00000001 with 0.1
        mu = torch.where(mu < 0.000001, torch.zeros_like(mu), mu)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class OntixTune(nn.Module):
    def __init__(self, trial, input_dim, cfg, latent_dim, mask_1, mask_2=None):
        super(OntixTune, self).__init__()
        if latent_dim == None:
            latent_dim = cfg["LATENT_DIM_FIXED"]
        self.latent_dim = latent_dim
        self.trial = trial
        self.input_dim = input_dim
        self.cfg = cfg
        dec_fc_layer = self.trial.suggest_int(
            "fc_n_layers", cfg["LAYERS_LOWER_LIMIT"], cfg["LAYERS_UPPER_LIMIT"]
        )
        if mask_2 == None:
            self.ont_dim_1 = mask_1.shape[1]
            self.ont_dim_2 = None
            self.n_layers = dec_fc_layer + 1
            ### Ignore fix latent dim if no additional fc layer specified for dimension reduction
            if dec_fc_layer == 0:
                self.latent_dim = self.ont_dim_1
        else:
            self.ont_dim_1 = mask_1.shape[1]
            self.ont_dim_2 = mask_2.shape[1]
            self.n_layers = dec_fc_layer + 2
            if dec_fc_layer == 0:
                self.latent_dim = self.ont_dim_2

        self.global_p = self.trial.suggest_float(
            f"dropout_all", cfg["DROPOUT_LOWER_LIMIT"], cfg["DROPOUT_UPPER_LIMIT"]
        )

        lower = 1
        upper = 64
        self.enc_factor = self.trial.suggest_float(f"encoding_factor", lower, upper)

        # Building architecture
        if self.ont_dim_2 == None:
            enc_dim = [input_dim, self.ont_dim_1]
        else:
            enc_dim = [input_dim, self.ont_dim_1, self.ont_dim_2]

        if dec_fc_layer > 0:
            enc_dim.extend(
                get_layer_dim(
                    enc_dim[-1], self.latent_dim, dec_fc_layer, self.enc_factor
                )
            )

        encoder_layers = []
        [
            encoder_layers.extend(FCLayer_1D(enc_dim[i], enc_dim[i + 1], self.global_p))
            for i in range(len(enc_dim) - 2)
        ]

        dec_dim = enc_dim[::-1]
        decoder_layers = []
        ## Ontology VAE only linear decoder
        [
            decoder_layers.extend(
                FCLayer_1D(dec_dim[i], dec_dim[i + 1], self.global_p, only_linear=True)
            )
            for i in range(len(dec_dim) - 1)
        ]

        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(enc_dim[-2], self.latent_dim)
        self.logvar = nn.Linear(enc_dim[-2], self.latent_dim)
        self.decoder = nn.Sequential(*decoder_layers)

        # Initialisierung
        self.apply(self._init_weights)
        # Decoder in Ontology VAE strictly positive for robust explaination
        self.decoder.apply(self._positive_dec)

        # Apply weight mask to create ontology-based decoder
        with torch.no_grad():
            self.decoder[-1].weight.mul_(mask_1)  ## Sparse Decoder Level 1
            if not mask_2 == None:
                self.decoder[-2].weight.mul_(mask_2)  ## Sparse Decoder Level 2

    def _positive_dec(self, m):
        if type(m) == nn.Linear:
            m.weight.data = m.weight.data.clamp(min=0)

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def encode(self, x):
        # latent = F.relu(self.encoder(x)) # ReLU before mu and logvar layer
        latent = self.encoder(x)  # ReLU already in encoder definition

        mu = self.mu(latent)
        logvar = self.logvar(latent)
        # prevent  mu and logvar from beeing to close to zero this increased
        # numerical stability
        logvar = torch.clamp(logvar, 0.1, 20)
        # replace mu when mu < 0.00000001 with 0.1
        mu = torch.where(mu < 0.000001, torch.zeros_like(mu), mu)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ImageVAETune(nn.Module):
    def __init__(self, trial, img_shape, latent_dim, cfg):
        super(ImageVAETune, self).__init__()
        self.trial = trial
        self.cfg = cfg
        self.nc, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.n_layers = self.trial.suggest_int(
            "fc_n_layers", cfg["LAYERS_LOWER_LIMIT"], cfg["LAYERS_UPPER_LIMIT"]
        )
        self.strides = 2
        self.padding = 1
        self.kernel_size = 4
        self.suggested_power = int(np.log2(self.h / 4))
        if cfg["TUNE_N_FILTERS"]:
            self.suggested_power = self.trial.suggest_int(
                "n_filters", 3, int(np.log2(self.h / 4))
            )
        self.n_filters = np.power(2, self.suggested_power)
        self.latent_filters = 0

        # ENCODER ------------------------------------------------------------
        encoder_layers = []
        for i in range(self.n_layers):
            cur_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.nc,
                    out_channels=self.n_filters,
                    kernel_size=self.kernel_size,
                    stride=self.strides,
                    padding=self.padding,
                ),
                nn.BatchNorm2d(self.n_filters),
                nn.LeakyReLU(0.2, inplace=True),
            )
            # no BatchNorm2D in first layer to match Uhler architecture
            if i == 0:
                cur_layer = nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.nc,
                        out_channels=self.n_filters,
                        kernel_size=self.kernel_size,
                        stride=self.strides,
                        padding=self.padding,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            if i == self.n_layers - 1:
                cur_layer = nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.nc,
                        out_channels=self.nc,
                        kernel_size=self.kernel_size,
                        stride=self.strides,
                        padding=self.padding,
                    ),
                    nn.BatchNorm2d(self.nc),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                encoder_layers.append(cur_layer)
                continue  # don't change number of channel after last layer
            encoder_layers.append(cur_layer)
            self.nc = copy.deepcopy(self.n_filters)
            self.n_filters *= 2
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_filters = copy.deepcopy(
            self.nc
        )  # number of filters for latent layers

        # to Calculate the image shape after the encoder, we need to know the number of layers
        # because the shape halfs after every Conv2D layer
        # So the output shape after all layers is in_shape / 2**N_layers
        # We showed above in the DocString why the shape halfs
        self.spatial_dim = self.h // (2**self.n_layers)
        # In the Linear mu and logvar layer we need to flatten the 3D output to a 2D matrix
        # Therefore we need to multiply the size of every out dimension of the input layer to the Linear layers
        # This is hidden_dim * 8 (the number of filter/channel layer) * spatial dim (the widht of the image) * spatial diem (the height of the image)
        # assuimg width = height
        # The original paper had a fixed spatial dimension of 2, which only worked for images with 64x64 shape

        self.mu = nn.Linear(
            self.latent_filters * self.spatial_dim * self.spatial_dim, self.latent_dim
        )
        self.logvar = nn.Linear(
            self.latent_filters * self.spatial_dim * self.spatial_dim, self.latent_dim
        )

        self.d1 = nn.Sequential(
            nn.Linear(
                self.latent_dim,
                self.latent_filters * self.spatial_dim * self.spatial_dim,
            ),
            nn.ReLU(inplace=True),
        )

        decoder_layers = []
        # DECODER -------------------------------------------------------------
        # to make first layer of decoder same size as last layer of encoder and in and outchannels same
        for i in range(self.n_layers):
            if i == 0:
                cur_layer = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=self.nc,
                        out_channels=self.nc,
                        kernel_size=self.kernel_size,
                        stride=self.strides,
                        padding=self.padding,
                    ),
                    nn.BatchNorm2d(self.nc),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                decoder_layers.append(cur_layer)
                continue
            if i == self.n_layers - 1:
                cur_layer = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=self.nc,
                        out_channels=self.img_shape[0],
                        kernel_size=self.kernel_size,
                        stride=self.strides,
                        padding=self.padding,
                    ),
                    nn.Sigmoid(),
                )
                decoder_layers.append(cur_layer)
                continue
            cur_layer = nn.Sequential(
                nn.ConvTranspose2d(
                    self.nc, self.nc // 2, self.kernel_size, self.strides, self.padding
                ),
                nn.BatchNorm2d(self.nc // 2),
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.nc //= 2

            decoder_layers.append(cur_layer)
        self.decoder = nn.Sequential(*decoder_layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def encode(self, x):
        h = self.encoder(x)
        # get view dimension

        h = h.view(-1, self.latent_filters * self.spatial_dim * self.spatial_dim)
        mu = self.mu(h)
        logvar = self.logvar(h)
        logvar = torch.clamp(logvar, 0.1, 20)
        mu = torch.where(mu < 0.00000001, torch.zeros_like(mu), mu)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.d1(z)
        # divide latent filters by 4 because we have 4 times the latent filters in last layer of encoder
        # expand byt 2,2 to match dimension
        # all this seems unitutuive, but we want to stay consistent with the Uhler papers
        h = h.view(-1, self.latent_filters, self.spatial_dim, self.spatial_dim)
        return self.decoder(h)

    def get_latent(self, x):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)

    def translate(self, z):
        out = self.decode(z)
        return out.view(self.img_shape)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out.view(x.shape), mu, logvar
