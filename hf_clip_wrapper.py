import torch
from PIL import Image
from transformers import AlignModel, AlignProcessor, CLIPModel, CLIPProcessor


class HFCLIPWrapper(object):
    def __init__(self, vlm_name, image_batch_size=32, text_batch_size=100) -> None:
        self.vlm_name = vlm_name
        self.model_type = "clip"
        if not self.vlm_name.startswith("openai"):
            self.model_type = "align"
        self.image_batch_size = image_batch_size
        self.text_batch_size = text_batch_size
        self.processor = None
        self.model = None
        self.load_processor()

    def supports_image_caching(self):
        return True

    def supports_text_caching(self):
        return True

    def load_processor(self):
        if self.processor is None:
            if self.model_type == "clip":
                self.processor = CLIPProcessor.from_pretrained(self.vlm_name)
            else:
                self.processor = AlignProcessor.from_pretrained(self.vlm_name)

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
        if load_in_16bit and self.model_type != "align":
            print("Loading VLM in 16bit")
            kwargs["torch_dtype"] = torch.float16
        if self.model_type == "clip":
            self.model = CLIPModel.from_pretrained(self.vlm_name, **kwargs)
        else:
            self.model = AlignModel.from_pretrained(self.vlm_name, **kwargs)
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()

    def model_device(self):
        assert self.model is not None
        return self.model.device

    def del_model(self):
        del self.model
        self.model = None

    def get_image_transform(self):
        image_transform = lambda ix: self.processor(
            images=ix, return_tensors="pt", padding=True, truncation=True
        )["pixel_values"][0]

        return image_transform

    def encode_text(self, texts, aggregate=False):
        with torch.no_grad():
            sub_embed_list = list()
            for i in range(0, len(texts), self.text_batch_size):
                subtext = texts[i : i + self.text_batch_size]
                tokenized_input = self.processor(
                    text=subtext,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                text_embeds = self.model.get_text_features(
                    **tokenized_input.to(self.model.device)
                )
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                sub_embed_list.append(text_embeds)

            text_embeds = torch.cat(sub_embed_list, dim=0)
            if aggregate:
                text_embeds = text_embeds.mean(dim=0)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds

    def encode_image(self, images):
        with torch.no_grad():
            sub_embed_list = list()
            for i in range(0, len(images), self.image_batch_size):
                sub_images = images[i : i + self.image_batch_size]
                if isinstance(sub_images[0], Image.Image):
                    sub_images = self.processor(images=sub_images, return_tensors="pt")[
                        "pixel_values"
                    ]
                    sub_images = sub_images.to(self.model.device)
                if self.model_type == "align":
                    sub_images = sub_images.to(
                        self.model.vision_model.embeddings.convolution.weight.dtype
                    )
                image_feats = self.model.get_image_features(pixel_values=sub_images)
                image_feats = image_feats / image_feats.norm(p=2, dim=-1, keepdim=True)
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
        for i in range(0, len(images), self.image_batch_size):
            sub_images = images[i : i + self.image_batch_size]
            sub_sims = sub_images @ texts.t()
            if self.model_type == "clip":
                sub_sims_list.append(sub_sims * self.model.logit_scale.exp())
            else:
                sub_sims_list.append(sub_sims / self.model.temperature)

        sims = torch.cat(sub_sims_list, dim=0)
        return sims
