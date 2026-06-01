from __future__ import annotations

import inspect
from typing import Any

from transformers import Trainer, TrainingArguments


def build_training_arguments(**kwargs: Any) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" in sig.parameters and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


def build_trainer(**kwargs: Any) -> Trainer:
    sig = inspect.signature(Trainer.__init__)
    tokenizer = kwargs.pop("tokenizer", None)
    if tokenizer is not None:
        if "processing_class" in sig.parameters:
            kwargs["processing_class"] = tokenizer
        elif "tokenizer" in sig.parameters:
            kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)
