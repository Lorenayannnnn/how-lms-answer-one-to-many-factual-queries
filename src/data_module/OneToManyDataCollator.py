from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Any

from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils import PaddingStrategy


@dataclass
class OneToManyDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {feature_name: [] for feature_name in features[0].keys()}
        name_of_features_to_be_padded = ['input_ids', 'attention_mask']

        features_to_be_padded = []

        for f in features:
            features_to_be_padded.append({feature_name: f.pop(feature_name) for feature_name in name_of_features_to_be_padded})
            for feature_name in f.keys():
                batch[feature_name].append(f[feature_name])

        padding_result = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features_to_be_padded,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch.update(padding_result)

        return batch