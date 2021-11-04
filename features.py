import pandas as pd
import numpy as np

# feature 1: convert POS-tags to integers
#This way we can train on it numerically
def pos_index(pos, pos_vocab):
  return pos_vocab.index(pos)

# feature 2: is this a proper noun?
def is_propn(pos):
  return pos=='PROPN'

# feature 3: is the first character a capital letter?
def title_case(tok):
  return tok[0:1].isupper()

# feature 4: is it likely an acronym?
def acronym(tok):
  return tok.isupper()

# feature 5: is it part of a hashtag?
def hashtag(tok):
  return "#" in tok

# feature 6: is it part of a @ tag?
def at(tok):
  return "@" in tok

# feature 7: is the word capitalised and a majority of the words around that word are not?
def capRatio(toks):

  isna = pd.isna(toks["token"])
  toks["isna"] = isna

  capitalised = [str(tok)[0:1].isupper() for tok in toks["token"]]
  toks["capitalised"] = capitalised

  isnaarr = np.array(isna)
  splitinxs = np.where(isnaarr)[0]
  capsplit = np.array(np.split(capitalised, splitinxs))

  numberWords = np.array([len(subarr) for subarr in capsplit])
  numberCapitalised = np.array([sum(subarr) for subarr in capsplit])

  ratios = numberCapitalised / numberWords

  capsplit[0] = np.concatenate([capsplit[0], [0]])

  zipratiocap = list(zip(capsplit, ratios))

  retval = np.array([np.full_like(cap[:-1], ratio) for (cap, ratio) in zipratiocap])

  retval = np.concatenate(retval)

  return retval


#feature 8: is the word in the wordlist dictionary?
#strange brand names etc. will not be in there, so it could be a good identifier of those words.
#it will however also pick up on hashtags and @s, hence the addition of those features.
def checkInDictionary(dict, tok):
  return tok in dict



# training feature
# training labels: convert BIO to integers
def bio_index(bio):
  ind = {"B": 0, "I": 1, "O": 2}
  return ind[bio]