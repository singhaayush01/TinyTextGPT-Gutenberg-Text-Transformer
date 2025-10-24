# step 2 : Train a BPE Tokenizer

# importing built-in and external libraries
import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# function to read all the downloaded text files
def get_dataset_text(data_dir="data/raw"):
    # reads every .txt file inside data/raw and returns a list of strings
    # each string contains the full content of one book
    texts = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

# create the tokenizer using BPE model
tokenizer = Tokenizer(models.BPE())

# define how to split the text before tokenizing
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

# define how to decode tokens back into text
tokenizer.decoder = decoders.ByteLevel()

# setup training configuration for our tokenizer
VOCAB_SIZE = 10000  # number of unique subword tokens to learn
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[pad]", "[eos]"],
    show_progress=True
)

# load text data from our dataset
text_data = get_dataset_text()

# train the tokenizer on our text/books
print("Training Tokenizer on Project Gutenberg dataset...")
tokenizer.train_from_iterator(text_data, trainer=trainer)

# enable padding so sequences have equal length when batching later
tokenizer.enable_padding(
    pad_id=tokenizer.token_to_id("[pad]"),
    pad_token="[pad]"
)

# save the trained tokenizer as a JSON file
tokenizer.save("gutenberg_tokenizer.json", pretty=True)
print("Tokenizer training complete! saved as 'gutenberg_tokenizer.json'")
