import pickle

from genre.fairseq_model import GENRE
from genre.trie import Trie

# load the prefix tree (trie)
with open("data/kilt_titles_trie_dict.pkl", "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))

# load the model
model = GENRE.from_pretrained("models/fairseq_entity_disambiguation_aidayago").eval()

# generate Wikipedia titles
model.sample(
    sentences=["Einstein was a [START_ENT] German [END_ENT] physicist."],
    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
)