
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
import nltk.corpus as corpus


# In[2]:


grame_2 = pd.read_fwf("data/w2_.txt",header=None)
grame_3 = pd.read_fwf("data/w3_.txt",header=None)
grame_4 = pd.read_fwf("data/w4_.txt",header=None)
grame_5 = pd.read_fwf("data/w5_.txt",header=None)


# In[3]:


def freq(words):
    words = words.split("\t")
    return words[0].lower()
def word1(words):
    words = words.split("\t")
    return words[1]
def word2(words):
    words = words.split("\t")
    if(len(words)>=3):
        return words[2].lower()
    else:
        return " " 
def word3(words):
    words = words.split("\t")
    if(len(words)>=4):
        return words[3].lower()
    else:
        return " " 

def word4(words):
    words = words.split("\t")
    if(len(words)>=5):
        return words[4].lower()
    else:
        return " " 

    
def word5(words):
    words = words.split("\t")
    if(len(words)>=6):
        return words[5].lower()
    else:
        return " " 


# In[142]:


all_words  = corpus.words.words()
def to_lower(word):
    return word.lower()
all_words = pd.DataFrame(all_words)[0].apply(to_lower)
all_words = pd.unique(all_words)
# grame_2[0] = grame_2[0].apply(to_lower)
# grame_3[0] = grame_3[0].apply(to_lower)
# grame_4[0] = grame_4[0].apply(to_lower)
# grame_5[0] = grame_5[0].apply(to_lower)


grame_2["freq"]  = grame_2[0].apply(freq)
grame_2["word1"] = grame_2[0].apply(word1)
grame_2["word2"] = grame_2[0].apply(word2)

grame_3["freq"]  = grame_3[0].apply(freq)
grame_3["word1"] = grame_3[0].apply(word1)
grame_3["word2"] = grame_3[0].apply(word2)
grame_3["word3"] = grame_3[0].apply(word3)

grame_4["freq"]  = grame_4[0].apply(freq)
grame_4["word1"] = grame_4[0].apply(word1)
grame_4["word2"] = grame_4[0].apply(word2)
grame_4["word3"] = grame_4[0].apply(word3)
grame_4["word4"] = grame_4[0].apply(word4)

grame_5["freq"]  = grame_5[0].apply(freq)
grame_5["word1"] = grame_5[0].apply(word1)
grame_5["word2"] = grame_5[0].apply(word2)
grame_5["word3"] = grame_5[0].apply(word3)
grame_5["word4"] = grame_5[0].apply(word4)
grame_5["word5"] = grame_5[0].apply(word5)



# In[143]:


def candidate(paragraph):
    words = paragraph.split(" ")
    flag_cond = 1
    if(len(words) == 5):
        word_candidate = (grame_5[(grame_5['word1'] == words[0]) & (grame_5['word2'] == words[1]) & (grame_5['word3'] == words[2])&(grame_5['word4'] == words[3])])['word5']
        if(len(word_candidate) > 0):
            flag_cond = 0
        else:
            words[1:]
    if(len(words) == 4 and flag_cond == 1):
        word_candidate = (grame_4[(grame_4['word1'] == words[0])&(grame_4['word2'] == words[1])&(grame_4['word3'] == words[2])])['word4']
        if(len(word_candidate) > 0):
            flag_cond = 0
        else:
            words[1:]            
    if(len(words) == 3 and flag_cond == 1):
        word_candidate = (grame_3[(grame_3['word1'] == words[0])&(grame_3['word2'] == words[1])])['word3']
        if(len(word_candidate) > 0):
            flag_cond = 0
        else:
            words[1:]
    if(len(words) == 2 and flag_cond == 1):
        word_candidate = (grame_2[(grame_2['word1'] == words[0])])['word2']
        if(len(word_candidate) > 0):
            flag_cond = 0
        else:
            words[1:]
    if(flag_cond == 1):
        word_candidate = all_words
        
    word_candidate = pd.DataFrame(word_candidate)
    word_candidate.columns = [0]
    if(len(word_candidate[word_candidate[0] == words[-1]]) > 0):
        return (words[-1],-1)
    else:
        return (words[-1],get_num_candidate(words[-1],word_candidate,5))
    


# In[144]:


def get_num_candidate(word,candidate,num):
    def get_distance(word_candidate):
        return nltk.edit_distance(word,word_candidate)
#     print(word)
    candidate['freq'] = candidate[0].apply(get_distance)
    return(candidate.sort_values(by="freq").head(num))
    


# In[145]:


a = candidate("salah go to schoo")


# In[146]:


if(type(a[-1]) == pd.core.frame.DataFrame):
    print(a[0])
    print(a[1])


# In[147]:


all_words

