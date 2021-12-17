# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers import RobertaTokenizerFast

class Feature:
    def __init__(self, idx, input_ids, input_mask, segment_ids, question_mask):
        self.idx = idx
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.question_mask = question_mask

    @classmethod
    def make_single(cls, idx, main_tokens, context_tokens, tokenizer, max_seq_length, vary_segment_id=False):
        main_tokens = main_tokens[-(max_seq_length-4):]
        tokens = [tokenizer.cls_token] + main_tokens + [tokenizer.sep_token] + context_tokens
        tokens = tokens[:max_seq_length-1]
        tokens = tokens + [tokenizer.sep_token]
        input_mask = [1.] * len(tokens)
        question_mask = [1.] * (1 + len(main_tokens)) + [0.] * (len(tokens) - len(main_tokens) - 1)
        if isinstance(tokenizer, RobertaTokenizerFast):
            segment_ids = [0] * len(tokens)
        elif vary_segment_id:
            segment_ids = [0] * (1 + len(main_tokens)) + [1] * (len(tokens) - len(main_tokens) - 1)
        else:
            segment_ids = [1] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        question_mask += padding
        segment_ids += padding

        return cls(idx, input_ids, input_mask, segment_ids, question_mask)

        
