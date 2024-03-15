import numpy as np
import torch


def cosine_reward(target_text_idx, similarity_matrix, image_classes, text_classes):
    target_txt_cls = text_classes[target_text_idx]
    cls_image_indices = torch.nonzero(image_classes == target_txt_cls, as_tuple=True)[0]
    target_similarities = similarity_matrix[cls_image_indices, :][:, target_text_idx]
    mean_similarities = target_similarities.mean()
    return mean_similarities


class ImageTextReward(object):
    def __init__(self, cache_image_embs=True, cache_text_embs=False) -> None:
        self.cache_image_embs = cache_image_embs
        self.cache_text_embs = cache_text_embs
        self.image_cache = dict()
        self.text_cache = dict()

    def get_cached_text(self, texts, vlm):
        if not self.cache_text_embs or not vlm.supports_text_caching():
            return texts
        missing_texts = list()
        for text in texts:
            if text not in self.text_cache:
                missing_texts.append(text)
        if len(missing_texts) != 0:
            missing_texts = list(set(missing_texts))
            with torch.no_grad():
                missing_embeds = vlm.encode_text(missing_texts)
            for i in range(len(missing_texts)):
                self.text_cache[missing_texts[i]] = missing_embeds[i].detach().cpu()

        text_embed_list = list()
        for text in texts:
            text_embed_list.append(self.text_cache[text].to(vlm.model_device()))
        text_embeds = torch.stack(text_embed_list)
        return text_embeds

    def get_cached_image(self, image_indices, vlm, image_dataset):
        if not self.cache_image_embs or not vlm.supports_image_caching():
            images_list = [image_dataset[i][0] for i in image_indices]
            return images_list
        missing_image_indices = list()
        for img_idx in image_indices:
            if img_idx not in self.image_cache:
                missing_image_indices.append(img_idx)
        if len(missing_image_indices) != 0:
            missing_image_indices = list(set(missing_image_indices))
            missing_images = [image_dataset[idx][0] for idx in missing_image_indices]
            with torch.no_grad():
                missing_embeds = vlm.encode_image(missing_images)
            for i in range(len(missing_image_indices)):
                self.image_cache[missing_image_indices[i]] = (
                    missing_embeds[i].detach().cpu()
                )

        image_embed_list = list()
        for img_idx in image_indices:
            image_embed_list.append(self.image_cache[img_idx].to(vlm.model_device()))
        image_embeds = torch.stack(image_embed_list)
        return image_embeds

    def get_response_reward(
        self, target_text_idx, similarity_matrix, image_classes, text_classes
    ):
        reward_kwargs = dict(
            target_text_idx=target_text_idx,
            similarity_matrix=similarity_matrix,
            image_classes=image_classes,
            text_classes=text_classes,
        )
        reward_ = cosine_reward(**reward_kwargs)
        return reward_

    def calc(
        self,
        queries,
        responses,
        query2metadata,
        img_dataset,
        per_cls_indices,
        vlm,
        dummy_rewards=False,
    ):
        if dummy_rewards:
            return np.random.uniform(10, 20, [len(responses)]).tolist()
        responses_list = list()
        responses_cls_list = list()
        for qr_idx in range(len(responses)):
            responses_list.append(responses[qr_idx])
            assert len(query2metadata[queries[qr_idx]]) == 1
            responses_cls_list.append(query2metadata[queries[qr_idx]][0])

        involved_classes = list(set(responses_cls_list))

        images_idx_list = list()
        images_cls_list = list()
        for cls in involved_classes:
            images_idx_list.extend(per_cls_indices[cls])
            images_cls_list.extend([cls] * len(per_cls_indices[cls]))

        all_texts = responses_list

        all_texts = self.get_cached_text(texts=all_texts, vlm=vlm)
        images_list = self.get_cached_image(
            image_indices=images_idx_list, vlm=vlm, image_dataset=img_dataset
        )
        with torch.no_grad():
            all_sims_matrix = vlm.get_score(images=images_list, texts=all_texts)
        device_ = all_sims_matrix.device

        images_cls_tensor = torch.tensor(images_cls_list, device=device_)

        responses_cls_tensor = torch.tensor(responses_cls_list, device=device_)
        reward_list = list()
        for response_idx in range(len(responses_list)):
            reward_list.append(
                self.get_response_reward(
                    target_text_idx=response_idx,
                    similarity_matrix=all_sims_matrix,
                    image_classes=images_cls_tensor,
                    text_classes=responses_cls_tensor,
                ).item()
            )
        return reward_list
