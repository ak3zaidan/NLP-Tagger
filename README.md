# NLP-Tagger
Glove Embedding + GRU + Torch. Takes an input string and classifies each token(word) into tags in a sentence, such as Noun, Verb, Pronoun, Adjectives and so on.

**Model is already pretrained and stored inside of `model.pt`**
**Vocab is already built and stored in vocab.pkl**

- To run simple call "python tag.py"
- Enter a sentence and press enter
- The model will automcatically tag each token and print the result.

# Examples

<img width="621" alt="Screenshot 2025-04-20 at 1 42 49 PM" src="https://github.com/user-attachments/assets/898ca557-0109-4e44-bbe6-1bd684c6b340" />

<img width="627" alt="Screenshot 2025-04-20 at 1 43 01 PM" src="https://github.com/user-attachments/assets/d46eade3-06f1-4bca-a9f5-bbe80c96a843" />

# Training

- Best success with Glove embeddings
- Achieved higher train acc and val than base

<img width="625" alt="Screenshot 2025-04-20 at 1 44 53 PM" src="https://github.com/user-attachments/assets/02e28702-3da0-4269-abbe-818e76e3e09d" />
