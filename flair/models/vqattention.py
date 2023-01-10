import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

import flair.nn
from flair.data import Dictionary, Label, Sentence, Span
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import StackedEmbeddings, TokenEmbeddings
from flair.file_utils import cached_path, unzip_file
from flair.training_utils import store_embeddings

from flair.models.sequence_tagger_utils.bioes import get_spans_from_bio
from flair.models.sequence_tagger_utils.crf import CRF
from flair.models.sequence_tagger_utils.viterbi import ViterbiDecoder, ViterbiLoss
from flair.models.iattention import IAttentionHead

log = logging.getLogger("flair")


class IAttentionHead(IAttentionHead):
    def __init__(self):
