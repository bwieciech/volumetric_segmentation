import torch
from torchviz import make_dot

from volseg.utils.image_dimension_wrapper import ImageDimensionsWrapper
from volseg.utils.io_utils import print_info_message
from volseg.utils.padding import calculate_required_paddings


class UNet3d(torch.nn.Module):
    def __init__(self, num_classes, image_dimensions=(3, 116, 132, 132)):
        """
        :param image_dimensions: (channels, depth, height, width) or ImageDimensionsWrapper
        :param num_classes: number of classes to segment (e.g. liver, pancreas, lung...)
        """

        super().__init__()
        self.num_classes = num_classes
        self.image_dimensions = ImageDimensionsWrapper(dims=image_dimensions)

        conv3d_transpose_paddings = calculate_required_paddings(
            *self.image_dimensions.get_dhw(), num_levels=4
        )
        self.layers = torch.nn.ModuleDict(
            {
                "encoder_level_1": UNet3d.__build_conv_block(
                    in_channels=self.image_dimensions.channels,
                    intermediate_out_channels=32,
                    out_channels=64,
                ),
                "encoder_level_2": UNet3d.__build_conv_block(
                    in_channels=64, intermediate_out_channels=64, out_channels=128
                ),
                "encoder_level_3": UNet3d.__build_conv_block(
                    in_channels=128, intermediate_out_channels=128, out_channels=256
                ),
                # The "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation" paper explicitly states
                # that it doesn't use bottleneck layers. This is where one would usually live, hence the name.
                "not_bottleneck": UNet3d.__build_conv_block(
                    in_channels=256, intermediate_out_channels=256, out_channels=512
                ),
                "decoder_level_3": UNet3d.__build_conv_block(
                    in_channels=256 + 512,
                    intermediate_out_channels=256,
                    out_channels=256,
                ),
                "decoder_level_2": UNet3d.__build_conv_block(
                    in_channels=128 + 256,
                    intermediate_out_channels=128,
                    out_channels=128,
                ),
                "decoder_level_1": UNet3d.__build_conv_block(
                    in_channels=64 + 128, intermediate_out_channels=64, out_channels=64
                ),
                "max_pool": torch.nn.MaxPool3d(
                    kernel_size=2, stride=2, return_indices=False
                ),
                "conv_3d_transpose_not_bottleneck": UNet3d.__build_conv3d_transpose(
                    512, output_padding=conv3d_transpose_paddings[4]
                ),
                "conv_3d_transpose_level_3": UNet3d.__build_conv3d_transpose(
                    256, output_padding=conv3d_transpose_paddings[3]
                ),
                "conv_3d_transpose_level_2": UNet3d.__build_conv3d_transpose(
                    128, output_padding=conv3d_transpose_paddings[2]
                ),
                "output_conv": torch.nn.Conv3d(
                    in_channels=64,
                    out_channels=self.num_classes,
                    kernel_size=3,
                    padding="same",
                ),
            }
        )

    def forward(self, x):
        (
            encoder_level_1_output,
            encoder_level_2_output,
            encoder_level_3_output,
            encoder_level_3_pooled,
        ) = self.__encode(x)

        not_bottleneck_output = self.layers["not_bottleneck"](encoder_level_3_pooled)
        not_bottleneck_upscaled = self.layers["conv_3d_transpose_not_bottleneck"](
            not_bottleneck_output
        )

        decoder_level_3_upscaled = self.__decode(
            encoder_level_3_output, not_bottleneck_upscaled, level=3
        )
        decoder_level_2_upscaled = self.__decode(
            encoder_level_2_output, decoder_level_3_upscaled, level=2
        )

        decoder_level_1_input = torch.concat(
            (encoder_level_1_output, decoder_level_2_upscaled), axis=1
        )
        decoder_level_1_output = self.layers["decoder_level_1"](decoder_level_1_input)

        return decoder_level_1_output

    def visualize(self):
        print_info_message(
            "In case of \"Not a directory: PosixPath('dot')\" error, install graphviz manually, "
            "e.g. through apt."
        )

        input = torch.zeros(
            1, *self.image_dimensions.get(), dtype=torch.float, requires_grad=False
        )
        output = self(input)
        return make_dot(
            output.mean(),
            params=dict(self.named_parameters()),
            show_attrs=True,
            show_saved=True,
        )

    def __encode(self, x):
        encoder_level_1_output = self.layers["encoder_level_1"](x)
        encoder_level_1_pooled = self.layers["max_pool"](encoder_level_1_output)
        encoder_level_2_output = self.layers["encoder_level_2"](encoder_level_1_pooled)
        encoder_level_2_pooled = self.layers["max_pool"](encoder_level_2_output)
        encoder_level_3_output = self.layers["encoder_level_3"](encoder_level_2_pooled)
        encoder_level_3_pooled = self.layers["max_pool"](encoder_level_3_output)
        return (
            encoder_level_1_output,
            encoder_level_2_output,
            encoder_level_3_output,
            encoder_level_3_pooled,
        )

    def __decode(
        self, encoder_level_n_output, decoder_level_n_minus_one_upscaled, level
    ):
        decoder_level_n_input = torch.concat(
            (encoder_level_n_output, decoder_level_n_minus_one_upscaled), axis=1
        )
        decoder_level_n_output = self.layers[f"decoder_level_{level}"](
            decoder_level_n_input
        )
        decoder_level_n_upscaled = self.layers[f"conv_3d_transpose_level_{level}"](
            decoder_level_n_output
        )
        return decoder_level_n_upscaled

    @staticmethod
    def __build_conv_block(in_channels, intermediate_out_channels, out_channels):
        layers = [
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=intermediate_out_channels,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.BatchNorm3d(num_features=intermediate_out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(
                in_channels=intermediate_out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.BatchNorm3d(num_features=out_channels),
            torch.nn.ReLU(),
        ]
        return torch.nn.Sequential(*layers)

    @staticmethod
    def __build_conv3d_transpose(channels, output_padding):
        return torch.nn.ConvTranspose3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            output_padding=output_padding,
        )
