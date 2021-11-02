from features import *
from loadData import *

def get_pos_vocab():
  train = loadData("train")
  pos_vocab = train.dropna().upos.unique().tolist()
  return pos_vocab

def get_dictionary():
  with open("words_alpha.txt") as data:
    dictionary = data.read()
    dictionary_format = set(dictionary.split("\n"))
    return dictionary_format

def getTextNA(txt):
  txtna = txt.reset_index(drop=True)
  return txtna

def getTextNoNA(txt):
  txtnona = txt.dropna()  # drop empty rows between texts
  return getTextNA(txtnona)

def prepareText(txt):
  # This function simply takes the txt file, finds all B and I tags, and takes segments up to 3 either side of these tags. N/A rows are then added.
  # This means that at most we have 7:1 ratio between B and I.
  bioIn = [(bio != "O") for bio in txt['bio_only']]
  bioSmear = np.convolve(bioIn, [True, True, True, True, True, True, True], mode="same")

  bioSmearSmooth = np.clip(bioSmear, False, True)
  txt['bi_true'] = bioSmearSmooth

  txt[txt['bi_true'] == False] = np.nan

  #Fixes issue where bi_true set to True even when NaN
  txt[txt['token'] == np.nan] = np.nan

  return txt.drop(['bi_true'], axis=1)


# pass a data frame through our feature extractor
def extract_features(txt, preparetxt=False):
  pos_vocab = get_pos_vocab()
  dictionary = get_dictionary()

  if preparetxt:
    txt = prepareText(txt)

  print(txt.head(75))

  txtna = getTextNA(txt)
  txtnona = getTextNoNA(txt)

  print(txtnona.head(50))

  posinds = [pos_index(u, pos_vocab) for u in txtnona['upos']]
  txtnona['pos_indices'] = posinds

  isprop = [is_propn(u) for u in txtnona['upos']]
  txtnona['is_propn'] = isprop

  tcase = [title_case(t) for t in txtnona['token']]
  txtnona['title_case'] = tcase

  tacr = [acronym(t) for t in txtnona['token']]
  txtnona['acronym'] = tacr

  thash = [hashtag(t) for t in txtnona['token']]
  txtnona['hashtag'] = thash

  tat = [at(t) for t in txtnona['token']]
  txtnona['at'] = tat

  tcapratio = capRatio(txtna[['token']])
  txtnona['capital_ratio'] = tcapratio

  tdict = [checkInDictionary(dictionary, t) for t in txtnona['token']]
  txtnona['dict'] = tdict

  bioints = [bio_index(b) for b in txtnona['bio_only']]
  txtnona['bio_only'] = bioints

  print(txtnona.head(10))

  return txtnona