# @Author       : Duhongkai
# @Time         : 2024/2/1 16:59
# @Description  : 数据整合

import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Any, Optional, Union

from transformers.utils import PaddingStrategy


# 由transformers\data\data_collator\DataCollatorForSeq2Seq改写而来
class DataCollatorForSeq2Seq:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 robertaTokenizer: PreTrainedTokenizerBase,
                 model: Optional[Any] = None,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 roberta_max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 label_pad_token_id: int = -100,
                 return_tensors: str = "pt",
                 ):
        self.tokenizer = tokenizer
        self.robertaTokenizer = robertaTokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.roberta_max_length = roberta_max_length

    def __call__(self, cur_dataset, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        # cur_dataset中包含一个batch的信息，先进行划分
        features = list()               # 大模型需要的数据
        doctor_features = list()        # doctor 表示
        patient_features = list()       # patient 表示
        for single_data in cur_dataset:
            features.append(single_data["full_prompt"])
            doctor_features.append(single_data["doctor"])
            patient_features.append(single_data["patient"])
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        doctor_features = self.robertaTokenizer.pad(
            doctor_features,
            padding=self.padding,
            max_length=self.roberta_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        patient_features = self.robertaTokenizer.pad(
            patient_features,
            padding=self.padding,
            max_length=self.roberta_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        res_data = {"full_prompt": features, "labels": features["labels"], "doctor": doctor_features, "patient": patient_features}
        return res_data
