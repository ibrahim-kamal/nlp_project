
# coding: utf-8

# In[22]:


import pandas as pd
import nltk
import nltk.corpus as corpus


# In[202]:


grame_2 = pd.read_fwf("data/w2_.txt",header=None)
grame_3 = pd.read_fwf("data/w3_.txt",header=None)
grame_4 = pd.read_fwf("data/w4_.txt",header=None)
grame_5 = pd.read_fwf("data/w5_.txt",header=None)
test_set = pd.read_fwf("test/wikipedia.txt",sep=": ",header=None,)


# In[24]:


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


# In[184]:


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



# In[26]:


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
    


# In[35]:


def get_num_candidate(word,candidate,num):
    def get_distance(word_candidate):
        return nltk.edit_distance(word,word_candidate)
#     print(word)
    candidate['distance'] = candidate[0].apply(get_distance)
    return(candidate.sort_values(by="distance").head(num))
    


# In[137]:


def gui_condiate(text):
    gui_text = {}
    words = text.split(' ')
    for i in range(0,len(words)):
        
        if(i < 4):
            text = " ".join(words[0:i+1])
        else:
            text = " ".join(words[i-4:i+1])
#         print(text)
        if(i in gui_text):
            if(gui_text[i] != words[i]):
                cand = candidate(text)
                if(type(cand[-1]) == pd.core.frame.DataFrame):
                    gui_text[i] = (cand[0],get_num_candidate(cand[0],cand[1],3)[0].values.tolist())
                else:
                    gui_text[i] = (cand[0],[])
        else:
            cand = candidate(text)
            if(type(cand[-1]) == pd.core.frame.DataFrame):
                gui_text[i] = (cand[0],get_num_candidate(cand[0],cand[1],3)[0].values.tolist())
            else:
                gui_text[i] = (cand[0],[])
                
    return gui_text


# In[132]:


def wrong_word(row):
    return row.split(':')[0].lower().strip()
def target(row):
    return row.split(':')[1].lower().strip().split(" ")


# In[205]:


test_set["wrong_word"] = test_set[0].apply(wrong_word)
test_set["target"] = test_set[0].apply(target)


# In[150]:


def test_gui_condiate(text):
    cand = candidate(text)
    if(type(cand[-1]) == pd.core.frame.DataFrame):
        return get_num_candidate(cand[0],cand[1],3)[0].values.tolist()
    else:
        return []


# In[207]:


def fun_select(word):
    return not(word in all_words)
a = test_set["wrong_word"].apply(fun_select)
test_set = test_set[a]


# In[214]:


test_set['y_hat'] = test_set["wrong_word"].apply(test_gui_condiate)

