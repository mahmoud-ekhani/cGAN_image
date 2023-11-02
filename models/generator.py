import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim, n_classes, layers_size=[64, 128, 256, 512], show_size=False):
        """
        Generator class for a Conditional Generative Adversarial Network (CGAN).
        
        Args:
        - latent_dim (int): The latent dimension.
        - n_classes (int): The number of classes for conditional generation.
        - layers_size (list of ints): List containing sizes for the layers of the network.
        - show_size (bool): If True, print the sizes of inputs and outputs of each layer.
        """
        super(Generator, self).__init__()

        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.layers_size = layers_size 
        self.label_emb = nn.Embedding(n_classes, self.latent_dim)
        self.combine = nn.Linear(2 * self.latent_dim, layers_size[0] * self.img_shape ** 2)

        # Encoder and Decoder layers
        self.encoder_layers = self._build_encoder()
        self.decoder_layers = self._build_decoder()

        # If show_size is True, register hooks to print the sizes
        if show_size:
            self._register_hooks(self, self._hook_fn)

    def _build_encoder(self):
        """Construct the encoder part of the generator."""
        encoder = nn.Sequential(
            self._encoder_block(self.layers_size[0], self.layers_size[1]),
            self._encoder_block(self.layers_size[1], self.layers_size[2]),
            self._encoder_block(self.layers_size[2], self.layers_size[3])
        )
        return encoder

    def _encoder_block(self, in_channels, out_channels, max_pooling=True):
        """A building block for the encoder."""
        if max_pooling:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool2d(2, 2)  # Reduces spatial size by half
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )


    def _build_decoder(self):
        """Construct the decoder part of the generator."""
        decoder = nn.Sequential(
            self._decoder_block(self.layers_size[3], self.layers_size[2]),
            self._decoder_block(self.layers_size[2], self.layers_size[1]),
            self._decoder_block(self.layers_size[1], self.layers_size[0]),
            self._decoder_block(self.layers_size[0], 1, kernel=3, stride=1, padding=1, batch_norm=False, last_layer=True)
        )
        return decoder

    def _decoder_block(self, in_channels, out_channels, kernel=4, stride=2, padding=1, batch_norm=True, last_layer=False):
        """A building block for the decoder."""
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if last_layer:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)


    def forward(self, z, labels):
        """
        Forward pass of the generator.

        Args:
        - z (torch.Tensor): Noise tensor.
        - labels (torch.Tensor): Labels tensor for conditional generation.

        Returns:
        - img (torch.Tensor): Generated image tensor.
        """
        labels = self.label_emb(labels)
        combined = torch.cat([z, labels], 1)
        x = self.combine(combined).view(combined.size(0), 
                                        self.layers_size[0], 
                                        self.img_shape,
                                        self.img_shape)

        # Encoding
        x = self.encoder_layers(x)

        # Decoding
        img = self.decoder_layers(x)

        return img
    
    def _hook_fn(self, module, input, output):
        """
        A function to be hooked onto a module. It prints the sizes of input and output tensors.
        """
        print(f"Layer: {module.__class__.__name__}")
        print(f"Input shape: {input[0].shape}")
        print(f"Output shape: {output.shape}\n")

    def _register_hooks(self, model, hook_fn):
        """
        Registers the hook function on every submodule of the provided model.
        """
        for layer in model.children():
            # If the layer is a Sequential module, register hook to each of its layers
            if isinstance(layer, nn.Sequential):
                for sub_layer in layer:
                    sub_layer.register_forward_hook(hook_fn)
            else:
                layer.register_forward_hook(hook_fn)



