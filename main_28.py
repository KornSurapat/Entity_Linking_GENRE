import pickle
import json

from genre.fairseq_model_14 import GENRE
from genre.trie import Trie

# load the prefix tree (trie)
with open("data/kilt_titles_trie_dict.pkl", "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))

# load the model
model = GENRE.from_pretrained("models/fairseq_entity_disambiguation_aidayago").eval()

# generate Wikipedia titles
def read_jsonl(file_path):
    test_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for l in f:
            json_object = json.loads(l.strip())
            test_data.append(json_object)
    return test_data

def transform_input(test_data, problem):
    sentences = []
    for mention in test_data[problem]["gold_spans"]:
        sentence = test_data[problem]["text"][ : mention["start"]] + " [START_ENT] " + test_data[problem]["text"][mention["start"] : mention["start"] + mention["length"]] + " [END_ENT] " + test_data[problem]["text"][mention["start"] + mention["length"] : -1]
        sentences.append(sentence)
    return sentences

def transform_output(added_data, problem, output):
    for mention in range(len(output)):
        record = []
        for candidate in range(len(output[mention])):
            record.append([output[mention][candidate]["text"], float(output[mention][candidate]["score"])])
        added_data[problem]["gold_spans"][mention]["predictions_genre"] = record
    return added_data

def write_jsonl(file_path, added_data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in added_data:
            f.write(json.dumps(item) + '\n')

def process(file_path_read, file_path_write):
  # Read data from file
  test_data = read_jsonl(file_path_read)
  # Prepare container
  added_data = test_data
  for i in range(len(test_data)):
    # Transform input
    sentences = transform_input(test_data, i)
    # Run model
    output = model.sample(
        sentences,
        prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
    )
    # Transform output
    added_data = transform_output(added_data, i, output)
  # Store data to file
  write_jsonl(file_path_write, added_data)

# Just test
# test_data = read_jsonl("ace2004.jsonl")
# added_data = test_data
# for i in range(len(test_data)):
#     sentences = transform_input(test_data, i)
#     output = model.sample(
#         sentences,
#         prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
#     )
#     added_data = transform_output(added_data, i, output)
#     # Show
#     for mention in output:
#         print(mention)
#         print("__________________________________________________")
#     print(str(i + 1) + " done ____________________________________________________________________________________________________")

process("ace2004.jsonl", "ace2004_pred.jsonl")
print("1 done ____________________________________________________________________________________________________")
process("aida.jsonl", "aida_pred.jsonl")
print("2 done ____________________________________________________________________________________________________")
process("aquaint.jsonl", "aquaint_pred.jsonl")
print("3 done ____________________________________________________________________________________________________")
process("cweb.jsonl", "cweb_pred.jsonl")
print("4 done ____________________________________________________________________________________________________")
process("graphq.jsonl", "graphq_pred.jsonl")
print("5 done ____________________________________________________________________________________________________")
process("mintaka.jsonl", "mintaka_pred.jsonl")
print("6 done ____________________________________________________________________________________________________")
process("msnbc.jsonl", "msnbc_pred.jsonl")
print("7 done ____________________________________________________________________________________________________")
process("reddit_comments.jsonl", "reddit_comments_pred.jsonl")
print("8 done ____________________________________________________________________________________________________")
process("reddit_posts.jsonl", "reddit_posts_pred.jsonl")
print("9 done ____________________________________________________________________________________________________")
process("shadow.jsonl", "shadow_pred.jsonl")
print("10 done ____________________________________________________________________________________________________")
process("tail.jsonl", "tail_pred.jsonl")
print("11 done ____________________________________________________________________________________________________")
process("top.jsonl", "top_pred.jsonl")
print("12 done ____________________________________________________________________________________________________")
process("tweeki.jsonl", "tweeki_pred.jsonl")
print("13 done ____________________________________________________________________________________________________")
process("webqsp.jsonl", "webqsp_pred.jsonl")
print("14 done ____________________________________________________________________________________________________")
process("wiki.jsonl", "wiki_pred.jsonl")
print("15 done ____________________________________________________________________________________________________")

# output = model.sample(
#     sentences=["[START_ENT]Einstein[END_ENT] was a German physicist.", "Einstein was a [START_ENT]German[END_ENT] physicist."],
#     prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
# )
# print(output)
# print("____________________________________________________________________________________________________")
# output_2 = model.sample(
#     sentences=["[START_ENT] Einstein [END_ENT] was a German physicist.", "Einstein was a [START_ENT] German [END_ENT] physicist."],
#     prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
# )
# print(output_2)