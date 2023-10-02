from torch import nn
from typing import Optional
from torch.nn.functional import dropout, dropout2d
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.encoders import get_encoder
#from deeplabv3plus_custom.encoders import get_encoder
#from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from deeplabv3plus_custom.decoder import DeepLabV3PlusDecoder
from torch import randn_like, randint
#from deeplabv3plus_custom.addnoise import AddNoise

class DeepLabV3Plus(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        noise_std: float = 2,
        dropout_prob = 0.5
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError("Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride))

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
    
    def enable_random(self, model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout') or m.__class__.__name__.startswith('AddNoise'):
                m.train()
    

    def random_forward(self, x):

        self.check_input_shape(x)
        features = self.encoder(x)

        # features_dropout = []
        # for feature in features:
        #     dropout_tmp = dropout(feature, p=self.dropout_prob) 
        #     noise_tmp = randn_like(feature) * self.noise_std
        #     tmp = dropout_tmp + noise_tmp
        #     features_dropout.append(tmp)
            #features_dropout.append(dropout_tmp)

        decoder_output = self.decoder(*features)
        # decoder_output = self.decoder(*features_dropout)
        # decoder_output = dropout(decoder_output)
        # gaussian_noise = randn_like(decoder_output) * self.noise_std
        # masks = self.segmentation_head(decoder_output + gaussian_noise)
        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
    
