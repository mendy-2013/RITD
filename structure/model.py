import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import backbones
import decoders
import clip
from PIL import Image
from clip import clip
from clip.clip.model import CLIP


class BasicModel(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)

        self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
        self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))

    # def forward(self, data, *args, **kwargs):
    def forward(self, data, aux_data, *args, **kwargs):   # ##3
        self.aux_data = aux_data

        # if not hasattr(self, 'image_features') or not hasattr(self, 'text_features'):
        #     device = "cuda" if torch.cuda.is_available() else "cpu"
        #     model, preprocess = clip.load("ViT-B/32", device=device)
        #
        #     image = preprocess(Image.open("CLIP.jpg")).unsqueeze(0).to(device)
        #     text = clip.tokenize(["text region", "non-text region"]).to(device)
        #
        #     with torch.no_grad():
        #         self.image_features = model.encode_image(image)
        #         # self.text_features = model.encode_text(text)
        #

        # return self.decoder(self.backbone(data), aux_data, self.text_features, *args, **kwargs)
        return self.decoder(self.backbone(data), aux_data, *args, **kwargs)
        # return self.decoder(self.backbone(data), *args, **kwargs)

def parallelize(model, distributed, local_rank):
    if distributed:
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=[local_rank],
            find_unused_parameters=True)
    else:
        return nn.DataParallel(model)

class SegDetectorModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        super(SegDetectorModel, self).__init__()
        from decoders.seg_detector_loss import SegDetectorLossBuilder

        self.model = BasicModel(args)
        # for loading models
        self.model = parallelize(self.model, distributed, local_rank)
        self.criterion = SegDetectorLossBuilder(
            args['loss_class'], *args.get('loss_args', []), **args.get('loss_kwargs', {})).build()
        self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return os.path.join('seg_detector', args['backbone'], args['loss_class'])

    def forward(self, batch, training=True):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
            aux_data = batch["gt"].to(self.device)  # ##1
            # a = torch.sum(aux_data == 1).item()
        else:
            data = batch.to(self.device)
        data = data.float()
        # aux_data = aux_data.float()

        print(data.device)
        # pred = self.model(data, training=self.training)
        pred = self.model(data, aux_data, training=self.training)  # ##2
        ###### 传数据

        if self.training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred