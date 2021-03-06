# reload training file
wnuttrain = 'https://storage.googleapis.com/wnut-2017_ner-shared-task/wnut17train_clean_tagged.txt'
train = pd.read_table(wnuttrain, header=None, names=['token', 'label', 'bio_only', 'upos']).dropna()  # drop empty rows

# in order to convert POS tags to integers: get the UPOS tagset
pos_vocab = train.upos.unique().tolist()

# feature 1: convert POS-tags to integers
def pos_index(pos):
  ind = pos_vocab.index(pos)
  return ind

# feature 2: is this a proper noun?
def is_propn(pos):
  resp = False
  if pos=='PROPN':
    resp = True
  return resp

# feature 3: is the first character a capital letter?
def title_case(tok):
  resp = False
  if tok[0:1].isupper():
    resp = True  # thanks Archie Barrett for spotting a typo here!
  return resp

#Let's try and segment each tweet
#then segment into sentences
#then used that to try and find capitalised names
#we could also use a dictionary to find things which are not words and tag
#we then use a weak classifier to weight each of these features, and return a probability of being a prop noun
#finally we

# training labels: convert BIO to integers
def bio_index(bio):
  if bio=='B':
    ind = 0
  elif bio=='I':
    ind = 1
  elif bio=='O':
    ind = 2
  return ind

# pass a data frame through our feature extractor
def extract_features(txt):
  txt.dropna(inplace=True)  # drop empty rows between texts
  txt_copy = txt.reset_index(drop=True)
  posinds = [pos_index(u) for u in txt_copy['upos']]
  txt_copy['pos_indices'] = posinds
  isprop = [is_propn(u) for u in txt_copy['upos']]
  txt_copy['is_propn'] = isprop
  tcase = [title_case(t) for t in txt_copy['token']]
  txt_copy['title_case'] = tcase
  bioints = [bio_index(b) for b in txt_copy['bio_only']]
  txt_copy['bio_only'] = bioints
  return txt_copy