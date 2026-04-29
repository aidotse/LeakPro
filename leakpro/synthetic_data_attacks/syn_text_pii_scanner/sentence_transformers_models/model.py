#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Module containing Sentence Transformer models."""
from sentence_transformers import SentenceTransformer

#Set model_sen_trans
model_sen_trans = SentenceTransformer(
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    tokenizer_kwargs={"clean_up_tokenization_spaces": True}
)
