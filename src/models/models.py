import torch
import torch.nn as nn
import torch.nn.functional as F


def FCLayer_1D(in_dim, out_dim, drop_p=0, only_linear=False):
    if not only_linear:
        fc_layer = [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(drop_p),
            nn.ReLU(),
        ]
    else:
        fc_layer = [nn.Linear(in_dim, out_dim)]
    return fc_layer


def get_layer_dim(feature_dim, latent_dim, n_layers, enc_factor):
    layer_dimensions = [feature_dim]

    for i in range(n_layers - 1):
        prev_layer_size = layer_dimensions[-1]
        next_layer_size = max(int(prev_layer_size / enc_factor), latent_dim)
        layer_dimensions.append(next_layer_size)

    layer_dimensions.append(latent_dim)
    return layer_dimensions


class Vanillix(nn.Module):
    """A class to define a Vanilla Autoencoder"""

    def __init__(self, input_dim, latent_dim, global_p=0.1):
        super(Vanillix, self).__init__()
        # Model attributes
        self.input_dim = input_dim
        assert input_dim > 4, "input_dim must be greater than 4"
        self.n_layers = 2
        self.global_p = global_p
        self.enc_factor = 4
        self.latent_dim = latent_dim

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


class Varix(nn.Module):
    """A class to define a VAE model"""

    def __init__(self, input_dim, latent_dim, global_p=0.1):
        super(Varix, self).__init__()
        # Model attributes
        self.input_dim = input_dim
        assert input_dim > 4, "input_dim must be greater than 4"
        self.n_layers = 2
        self.global_p = global_p
        self.enc_factor = 4
        self.latent_dim = latent_dim

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


class Stackix(nn.Module):
    """A class to define a stacked or hierarchical Autoencoder"""

    def __init__(self, input_dim, latent_dim, global_p=0.1):
        super(Stackix, self).__init__()
        # Model attributes
        self.input_dim = input_dim
        assert input_dim > 4, "input_dim must be greater than 4"
        self.n_layers = 2
        self.global_p = global_p
        self.enc_factor = 4
        self.latent_dim = latent_dim

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


class Ontix(nn.Module):
    """A class to define a ontology-based VAE"""

    def __init__(
        self, input_dim, latent_dim, dec_fc_layer, mask_1, mask_2=None, global_p=0.1
    ):
        super(Ontix, self).__init__()
        # Model attributes
        self.input_dim = input_dim

        assert input_dim > 4, "input_dim must be greater than 4"

        self.latent_dim = latent_dim
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

        self.global_p = global_p
        self.enc_factor = 4

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


class ImageVAE(nn.Module):
    """
    This class defines a VAE, based on a CNN for images
    It takes as input an image and of shape (C,W,H) and reconstructs it.
    We ensure to have a latent space of shape <batchsize,1,LatentDim> and img_in.shape = img_out.shape
    We have a fixed kernel_size=4, padding=1 and stride=2 (given from https://github.com/uhlerlab/cross-modal-autoencoders/tree/master)

    So we need to calculate how the image dimension changes after each Convolution (we assume W=H)
    Applying the formular:
        W_out = (((W - kernel_size + 2padding)/stride) + 1)
    We get:
        W_out = (((W-4+2*1)/2)+1) =
        = (W-2/2)+1 =
        = (2(0.5W-1)/2) +1 # factor 2 out
        = 0.5W - 1 + 1
        W_out = 0.5W
    So in this configuration the output shape halfs after every convolutional step (assuming W=H)

    """

    def __init__(self, img_shape, latent_dim, hidden_dim=16):
        super(ImageVAE, self).__init__()
        self.nc, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.apply(self._init_weights)

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.nc,
                out_channels=self.hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.hidden_dim * 2,
                out_channels=self.hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.hidden_dim * 4,
                out_channels=self.hidden_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.hidden_dim * 8,
                out_channels=self.hidden_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # to Calculate the image shape after the encoder, we need to know the number of layers
        # because the shape halfs after every Conv2D layer
        self.num_encoder_layers = sum(
            1 for _ in self.encoder.children() if isinstance(_, nn.Conv2d)
        )
        # So the output shape after all layers is in_shape / 2**N_layers
        # We showed above in the DocString why the shape halfs
        self.spatial_dim = self.h // (2**self.num_encoder_layers)
        # In the Linear mu and logvar layer we need to flatten the 3D output to a 2D matrix
        # Therefore we need to multiply the size of every out diemension of the input layer to the Linear layers
        # This is hidden_dim * 8 (the number of filter/channel layer) * spatial dim (the widht of the image) * spatial diem (the height of the image)
        # assuimg width = height
        # The original paper had a fixed spatial dimension of 2, which only worked for images with 64x64 shape
        self.mu = nn.Linear(
            hidden_dim * 8 * self.spatial_dim * self.spatial_dim, self.latent_dim
        )
        self.logvar = nn.Linear(
            hidden_dim * 8 * self.spatial_dim * self.spatial_dim, self.latent_dim
        )

        # the same logic goes for the first decoder layer, which takes the latent_dim as inshape
        # which is the outshape of the previous mu/logvar layer
        # and the shape of the first ConvTranspose2D layer is the last outpus shape of the encoder layer
        # This the same multiplication as above
        self.d1 = nn.Sequential(
            nn.Linear(
                self.latent_dim, hidden_dim * 8 * self.spatial_dim * self.spatial_dim
            ),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * 8,
                out_channels=self.hidden_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * 8,
                out_channels=self.hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * 4,
                out_channels=self.hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * 2,
                out_channels=self.hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim,
                out_channels=self.nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _get_spatial_dim(self):
        return self.spatial_dim

    def encode(self, x):
        h = self.encoder(x)
        # this makes sure we get the <batchsize, 1, latent_dim> shape for our latent space in the next step
        # because we put all dimensionaltiy in the second dimension of the output shape.
        # By covering all dimensionality here, we are sure that the rest is
        h = h.view(-1, self.hidden_dim * 8 * self.spatial_dim * self.spatial_dim)
        logvar = self.logvar(h)
        mu = self.mu(h)
        # prevent  mu and logvar from being too close to zero, this increased
        # numerical stability
        logvar = torch.clamp(logvar, 0.1, 20)
        # replace mu when mu < 0.00000001 with 0.1
        mu = torch.where(mu < 0.000001, torch.zeros_like(mu), mu)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_latent(self, x):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)

    def decode(self, z):
        h = self.d1(z)
        # here we do a similar thing as in the encoder,
        # but instead of ensuring the correct dimension for the latent space,
        # we ensure the correct dimension for the first Conv2DTranspose layer
        # so we make sure that the last 3 dimension are (n_filters, reduced_img_dim, reduced_img_dim)
        h = h.view(-1, self.hidden_dim * 8, self.spatial_dim, self.spatial_dim)
        return self.decoder(h)

    def translate(self, z):
        out = self.decode(z)
        return out.view(self.img_shape)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out.view(x.shape), mu, logvar


class TranslateVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=1024):
        super(TranslateVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

        self.mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar = nn.Linear(hidden_dim, self.latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructions = self.decode(z)
        return reconstructions, mu, logvar


class LatentSpaceClassifier(nn.Module):
    """Latent space discriminator"""

    def __init__(self, input_dim, n_hidden=64, n_out=2):
        super(LatentSpaceClassifier, self).__init__()
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.ReLU(inplace=True),
            # nn.Linear(n_hidden, n_hidden),
            # nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)
