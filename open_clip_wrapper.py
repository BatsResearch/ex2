import gc

import open_clip
import torch
from PIL import Image

OCLIP_HF_HUB_PREFIX = "hf-hub:"
AMD_OPEN_CLIP_PREFIX = "r-open-clip:"


class OpenCLIPWrapper(object):
    def __init__(self, vlm_name, image_batch_size=32, text_batch_size=100) -> None:
        assert vlm_name.lower().startswith(AMD_OPEN_CLIP_PREFIX.lower())
        vlm_name = vlm_name[len(AMD_OPEN_CLIP_PREFIX) :]

        if vlm_name.startswith(OCLIP_HF_HUB_PREFIX):
            model_name = vlm_name
            pretrained = None
        else:
            model_name, pretrained = vlm_name.strip().split(":")
        self.model_name = model_name
        self.pretrained = pretrained

        self.image_batch_size = image_batch_size
        self.text_batch_size = text_batch_size
        self.processor = None
        self.model = None
        self.tokenizer = None

    def supports_image_caching(self):
        return True

    def supports_text_caching(self):
        return True

    def load_processor(self):
        if self.tokenizer is None:
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
        if self.processor is None:
            kwargs = dict()
            if self.pretrained is not None:
                kwargs["pretrained"] = self.pretrained
            _, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, **kwargs
            )
            gc.collect()
            self.processor = preprocess

    def load_model(self, device=None, load_in_16bit=False):
        assert load_in_16bit is not None
        if isinstance(load_in_16bit, str):
            if load_in_16bit.lower() == "false":
                load_in_16bit = False
            elif load_in_16bit.lower() == "true":
                load_in_16bit = True
            else:
                raise ValueError

        kwargs = dict()
        if self.pretrained is not None:
            kwargs["pretrained"] = self.pretrained
        if load_in_16bit:
            print("Loading VLM in 16bit")
            kwargs["precision"] = "fp16"
        if device is not None:
            kwargs["device"] = device

        model, _, _ = open_clip.create_model_and_transforms(self.model_name, **kwargs)
        self.model = model

    def model_device(self):
        assert self.model is not None
        return next(iter(self.model.parameters())).device

    def del_model(self):
        del self.model
        self.model = None

    def get_image_transform(self):
        assert self.processor is not None
        return self.processor

    def encode_text(self, texts, aggregate=False):
        with torch.no_grad(), torch.cuda.amp.autocast():
            sub_embed_list = list()
            for i in range(0, len(texts), self.text_batch_size):
                subtext = texts[i : i + self.text_batch_size]
                tokenized_input = self.tokenizer(subtext).to(self.model_device())
                text_embeds = self.model.encode_text(tokenized_input)
                text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
                sub_embed_list.append(text_embeds)

            text_embeds = torch.cat(sub_embed_list, dim=0)
            if aggregate:
                text_embeds = text_embeds.mean(dim=0)
                text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def encode_image(self, images):
        with torch.no_grad(), torch.cuda.amp.autocast():
            sub_embed_list = list()
            for i in range(0, len(images), self.image_batch_size):
                sub_images = images[i : i + self.image_batch_size]
                if isinstance(sub_images[0], Image.Image):
                    sub_images = torch.stack(
                        [self.processor(img) for img in sub_images]
                    )
                    sub_images = sub_images.to(self.model_device())
                image_feats = self.model.encode_image(sub_images)
                image_feats /= image_feats.norm(dim=-1, keepdim=True)

                sub_embed_list.append(image_feats)
            image_feats = torch.cat(sub_embed_list, dim=0)
        return image_feats

    def get_score(self, images, texts):
        if isinstance(images[0], Image.Image) or (
            isinstance(images, torch.Tensor) and images.dim() > 2
        ):
            images = self.encode_image(images)

        if isinstance(texts[0], str):
            texts = self.encode_text(texts)

        sub_sims_list = list()
        logits_scale = 100.0
        for i in range(0, len(images), self.image_batch_size):
            sub_images = images[i : i + self.image_batch_size]
            sub_sims = sub_images @ texts.T
            sub_sims = sub_sims * logits_scale
            sub_sims_list.append(sub_sims)

        sims = torch.cat(sub_sims_list, dim=0)
        return sims
