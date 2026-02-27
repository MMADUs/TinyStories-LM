# 1. find a text corpus with an average of 10-50M tokens (total token = avg row token * total row in a split)
# 2. rebuild torch dataset and preprocessing for corpus (just feed the training loop the text)
# 3. make sure that training loop will shift the target once using the original text (next token prediction)
# 4. i need to figure out how the inference result would looked like