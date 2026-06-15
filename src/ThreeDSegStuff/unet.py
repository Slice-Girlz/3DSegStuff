from typing import Literal, Optional, Tuple

import torch


class ConvBlock(torch.nn.Module):
    """A convolution block for a U-Net. Contains two convolutions, each followed by a ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Literal["same", "valid"] = "same",
        ndim: Literal[2, 3] = 2,
    ):
        """
        Args:
            in_channels (int): The number of input channels for this conv block.
            out_channels (int): The number of output channels for this conv block.
            kernel_size (int): The size of the kernel. A kernel size of N signifies an NxN or NxNxN
                kernel for ``ndim=2`` and ``ndim=3``, respectively.
            padding (Literal["same", "valid"], optional): The type of padding to
                use. "same" means padding is added to preserve the input dimensions.
                "valid" means no padding is added. Defaults to "same".
            ndim (Literal[2, 3], optional): Number of dimensions for the convolution operation. Use
                2 for 2D convolutions and 3 for 3D convolutions. Defaults to 2.

        Raises:
            ValueError: If unsupported values are used for padding or ndim.
        """
        super().__init__()
        if padding not in ("valid", "same"):
            msg = f"Invalid string value for padding: {padding=}. Options are same or valid."
            raise ValueError(msg)
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        convops = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        # define layers in conv pass
        self.conv_pass = torch.nn.Sequential(
            convops[ndim](
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
            convops[ndim](
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
        )

        for _name, layer in self.named_modules():
            if isinstance(layer, tuple(convops.values())):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.conv_pass(x)
        return output


class Downsample(torch.nn.Module):
    """Downsample module for U-Net"""

    def __init__(self, downsample_factor: int, ndim: Literal[2, 3] = 2):
        """
        Args:
            downsample_factor (int): Factor by which to downsample featuer maps.
            ndim (Literal[2,3], optional): Number of dimensions for the downsample operation.
                Defaults to 2.

        Raises:
            ValueError: If unsupported value is used for ndim.
        """

        super().__init__()
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        self.ndim = ndim
        downops = {2: torch.nn.MaxPool2d, 3: torch.nn.MaxPool3d}
        self.downsample_factor = downsample_factor

        self.down = downops[ndim](downsample_factor)

    def check_valid(self, image_size: Tuple[int, ...]) -> bool:
        """Check if the downsample factor evenly divides each image dimension."""
        for dim in image_size:
            if dim % self.downsample_factor != 0:
                return False
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor.

        Raises:
            RuntimeError: If shape of input is not divisible by downsampling factor.

        Returns:
            torch.Tensor: Downsampled tensor.
        """
        if not self.check_valid(tuple(x.size()[-self.ndim :])):
            raise RuntimeError(
                f"Can not downsample shape {x.size()} with factor {self.downsample_factor}"
            )
        output: torch.Tensor = self.down(x)
        return output


def center_crop(x, target_spatial_shape):
    """Center-crop x to match spatial dimensions given by target_spatial_shape."""

    x_target_size = x.size()[:2] + torch.Size(target_spatial_shape)

    offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

    slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

    return x[slices]


class CropAndConcat(torch.nn.Module):

    def forward(
        self, encoder_output: torch.Tensor, upsample_output: torch.Tensor
    ) -> torch.Tensor:
        encoder_cropped = center_crop(encoder_output, upsample_output.size()[2:])

        return torch.cat([encoder_cropped, upsample_output], dim=1)


class OutputConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[torch.nn.Module] = None,
        ndim: Literal[2, 3] = 2,
    ):
        """A convolutional block that applies a torch activation function.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            activation (torch.nn.Module  |  None, optional): An instance of any torch activation
                function (e.g., ``torch.nn.ReLU()``). Defaults to None for no activation after the
                convolution.
            ndim (Literal[2,3], optional): Number of dimensions for the convolution operation.
                Defaults to 2.
        Raises:
            ValueError: If unsupported values is used for ndim.
        """
        super().__init__()
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        convops = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        self.final_conv = convops[ndim](
            in_channels, out_channels, 1, padding=0
        )  # leave this out
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.final_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int = 1,   # affinities c = 3
        final_activation: Optional[torch.nn.Module] = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: Literal["same", "valid"] = "same",
        upsample_mode: str = "nearest",
        ndim: Literal[2, 3] = 2,
    ):
        """A U-Net for 2D or 3D input that expects tensors shaped like:
        ``(batch, channels, height, width)`` or ``(batch, channels, depth, height, width)``,
        respectively.

        Args:
            depth (int): The number of levels in the U-Net. 2 is the smallest that really makes
                sense for the U-Net architecture.
            in_channels (int): The number of input channels in the images.
            out_channels (int, optional): How many channels the output should have. Depends on your
                task. Defaults to 1.
            final_activation (Optional[torch.nn.Module], optional): Activation to use in final
                output block. Depends on your task. Defaults to None.
            num_fmaps (int, optional): Number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (int, optional): Factor by which to multiply the number of feature maps
                between levels. Level ``l`` will have ``num_fmaps*fmap_inc_factor**l`` feature maps.
                Defaults to 2.
            downsample_factor (int, optional): Factor for down- and upsampling of the feature maps
                between levels. Defaults to 2.
            kernel_size (int, optional): Kernel size to use in convolutions on both sides of the
                UNet. Defaults to 3.
            padding (Literal["same", "valid"], optional): The type of padding to
                use. "same" means padding is added to preserve the input dimensions.
                "valid" means no padding is added. Defaults to "same".
            upsample_mode (str, optional): The upsampling mode to pass to ``torch.nn.Upsample``.
                Usually "nearest" or "bilinear". Defaults to "nearest".
            ndim (Literal[2, 3], optional): Number of dimensions for the UNet. Use 2 for 2D-UNet and
                3 for 3D-UNet. Defaults to 2.

        Raises:
            ValueError: If unsupported values are used for padding or ndim.
        """

        super().__init__()
        if padding not in ("valid", "same"):
            msg = f"Invalid string value for padding: {padding=}. Options are same or valid."
            raise ValueError(msg)
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode

        # left convolutional passes
        self.left_convs = torch.nn.ModuleList()
        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            self.left_convs.append(
                ConvBlock(
                    fmaps_in, fmaps_out, self.kernel_size, self.padding, ndim=ndim
                )
            )

        # right convolutional passes
        self.right_convs = torch.nn.ModuleList()
        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.compute_fmaps_decoder(level)
            self.right_convs.append(
                ConvBlock(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding,
                    ndim=ndim,
                )
            )

        self.downsample = Downsample(self.downsample_factor, ndim=ndim)
        self.upsample = torch.nn.Upsample(
            scale_factor=self.downsample_factor,
            mode=self.upsample_mode,
        )
        self.crop_and_concat = CropAndConcat()
        self.final_conv = OutputConv(
            self.compute_fmaps_decoder(0)[1],
            self.out_channels,
            self.final_activation,
            ndim=ndim,
        )

    def compute_fmaps_encoder(self, level: int) -> Tuple[int, int]:
        """Compute the number of input and output feature maps for
        a conv block at a given level of the UNet encoder (left side).

        Args:
        ----
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        if level == 0:  # Leave out function
            fmaps_in = self.in_channels
        else:
            fmaps_in = self.num_fmaps * self.fmap_inc_factor ** (level - 1)

        fmaps_out = self.num_fmaps * self.fmap_inc_factor**level
        return fmaps_in, fmaps_out

    def compute_fmaps_decoder(self, level: int) -> Tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block
        at a given level of the UNet decoder (right side). Note:
        The bottom layer (depth - 1) is considered an "encoder" conv pass,
        so this function is only valid up to depth - 2.

        Args:
        ----
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        fmaps_out = self.num_fmaps * self.fmap_inc_factor ** (
            level
        )  # Leave out function
        concat_fmaps = self.compute_fmaps_encoder(level)[
            1
        ]  # The channels that come from the skip connection
        fmaps_in = concat_fmaps + self.num_fmaps * self.fmap_inc_factor ** (level + 1)

        return fmaps_in, fmaps_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # left side
        convolution_outputs = []
        layer_input = x
        for i in range(self.depth - 1):  # leave out center of for loop
            conv_out = self.left_convs[i](layer_input)
            convolution_outputs.append(conv_out)
            downsampled = self.downsample(conv_out)
            layer_input = downsampled

        # bottom
        conv_out = self.left_convs[-1](layer_input)
        layer_input = conv_out
        # right
        for i in range(0, self.depth - 1)[::-1]:  # leave out center of for loop
            upsampled = self.upsample(layer_input)
            concat = self.crop_and_concat(convolution_outputs[i], upsampled)
            conv_output = self.right_convs[i](concat)
            layer_input = conv_output
        if self.padding == "valid":
            lowest_res = self.downsample_factor ** (self.depth - 1)
            valid_shape = [dim - dim % lowest_res for dim in layer_input.size()[2:]]
            layer_input = center_crop(layer_input, valid_shape)
            output: torch.Tensor = self.final_conv(layer_input)
        else:
            output: torch.Tensor = self.final_conv(layer_input)
        return output