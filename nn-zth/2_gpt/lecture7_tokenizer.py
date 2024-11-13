# %%
# Byte-pair encoding algorithm for tokenization!

# Basic idea: we cannot use raw utf8 bytes for text prediction, mostly because
# the size of our attention head limits us in the amount of text that we can keep
# as context - if the context embeddings are single chars the span will be very small!

source_text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."

tokens = source_text.encode("utf-8")  # raw bytes
tokens = list(map(int, tokens))  # convert to list of integers

# Complex characters are encoded with multiple bytes, so token length will be higher!
print("text length: ", len(source_text))
print("token_length: ", len(tokens))

# %%
# The algorithm will search for the most common doublets and replace them with a single new number!


def find_most_common(tokens):
    counts = {}
    for key in zip(tokens[:-1], tokens[1:]):
        counts[key] = counts.get(key, 0) + 1

    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]


counts = find_most_common(tokens)
counts
# %%

# %%
