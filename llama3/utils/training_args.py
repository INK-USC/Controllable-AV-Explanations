from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """


    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )

    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear",
        metadata={"help": f"Which lr scheduler to use"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    preproc_data_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory that stores the preprocessed datasets."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    story_column: Optional[str] = field(
        default='story',
        metadata={"help": "The name of the column in the datasets containing stories."},
    )
    rationale_column: Optional[str] = field(
        default='rationale',
        metadata={"help": "The name of the column in the datasets containing the ground truth rationales."},
    )
    aspect_column: Optional[str] = field(
        default='aspect_string',
        metadata={"help": "The name of the column in the datasets containing the aspects for evaluation."},
    )
    rating_column: Optional[str] = field(
        default='rating',
        metadata={"help": "The name of the column in the datasets containing the ratings of different aspects."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=300,
        #default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        #default=128,
        default=64,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    predict_times: Optional[int] = field(
        default=3, # SAHANA
        metadata={
            "help": (
                "Number of output generation during prediction"
            )
        },
    )
    
    predict_file_name: Optional[str] = field(
        default="generated_predictions.txt", metadata={"help": "output name for do_predict"}
    )
    gen_top_k: Optional[int] = field(
        default=50,
        metadata={
            "help": (
                "generation top k value"
            )
        },
    )

    gen_top_p: Optional[float] = field(
        default=0.95,
        metadata={
            "help": (
                "generation top p value"
            )
        },
    )

    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    all_aspects: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether the training data has all aspects present."
            )
        },
    )

    no_aspects: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether the training data has no aspects present."
            )
        },
    )

    generated_file_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path where generated_rationales.txt is located"
            )
        },
    )

    with_story: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether story needs to be added in the scorer."
            )
        },
    )

    with_rationale: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether rationale needs to be added in the scorer."
            )
        },
    )

    i_or: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use I-OR like training."
            )
        },
    )

    i_ro: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use Chain of Thought like training."
            )
        },
    )

    teacher_forcing: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use Teacher Forcing like training."
            )
        },
    )

    anon_aspects: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to anonymise aspects"
            )
        },
    )

    rl_output: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether rationales are generated from RL-like training"
            )
        },
    )

    reranker: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether rationales are generated from reranker"
            )
        },
    )

    keywords: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether keywords are used as rationales"
            )
        },
    )

    keycot: Optional[str] = field(
        default="", metadata={"help": "which splitted keycot rationales to use."}
    )

    completion: Optional[str] = field(
        default="first", metadata={"help": "which splitted keycot rationales to use."}
    )


    #num_train_epochs: Optional[float] = field(
    #    default=3.0,
    #    metadata={
    #        "help": "num_train_epochs"
    #    },
    #)

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a training/validation/test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


