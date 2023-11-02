import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_shape, n_classes, layers_size=[128, 256, 512, 1024], show_size=False):
        """
        Discriminator class for a Conditional Generative Adversarial Network (CGAN).
        
        Args:
        - img_shape (int): The size of the input images.
        - n_classes (int): The number of classes for conditional discrimination.
        - layers_size (list of ints): List containing sizes for the layers of the network.
        - show_size (bool): If True, print the sizes of inputs and outputs of each layer.
        """
        super(Discriminator, self).__init__()

        self.img_shape = img_shape
        self.latent_dim = img_shape ** 2
        self.layers_size = layers_size
        self.label_emb = nn.Embedding(n_classes, self.latent_dim)
        self.combine = nn.Conv2d(2, layers_size[0], 3, stride=1, padding=1)

        # Discriminator layers
        self.discriminator_layers = self._build_discriminator()

        # Final layer
        self.fc = nn.Linear(layers_size[-1] * img_shape // 8 * img_shape // 8, 1)

        # If show_size is True, register hooks to print the sizes
        if show_size:
            self._register_hooks(self, self._hook_fn)

    def _build_discriminator(self):
        """Construct the main body of the discriminator."""
        discriminator = nn.Sequential(
            self._discriminator_block(self.layers_size[0], self.layers_size[1]),
            self._discriminator_block(self.layers_size[1], self.layers_size[2]),
            self._discriminator_block(self.layers_size[2], self.layers_size[3])
        )
        return discriminator

    def _discriminator_block(self, in_channels, out_channels):
        """A building block for the discriminator."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, img, labels):
        """
        Forward pass of the discriminator.

        Args:
        - img (torch.Tensor): Image tensor.
        - labels (torch.Tensor): Labels tensor for conditional discrimination.

        Returns:
        - validity (torch.Tensor): Validity tensor indicating real or fake.
        """
        labels = self.label_emb(labels).view(img.size(0), 1, self.img_shape, self.img_shape)
        combined = torch.cat([img, labels], 1)

        # Discrimination
        x = self.combine(combined)
        x = self.discriminator_layers(x)
        
        # Flatten and apply sigmoid
        x = x.view(x.size(0), -1)
        validity = torch.sigmoid(self.fc(x))
        return validity
    
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
