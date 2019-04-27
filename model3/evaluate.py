import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys 
sys.path.insert(0,'E:\\level4\\second\\NLP\\project\\deep-spell-checkr-master\\')
from utils import CharacterTable, transform
from utils import restore_model, decode_sequences
from utils import read_text, tokenize

error_rate = 0.6
reverse = True
model_path = './models/seq2seq.h5'
hidden_size = 512
sample_mode = 'argmax'
data_path = './data'
books = ['nietzsche.txt', 'pride_and_prejudice.txt', 'shakespeare.txt', 'war_and_peace.txt']

#test_sentence = 'The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.'
#test_sentence ='i have to gone to schoooll bot i am taired'
import os
import unidecode
def read_text(data_path, list_of_books):
    text = ''
    for book in list_of_books:
        file_path = os.path.join(data_path, book)
        strings = unidecode.unidecode(open(file_path).read())
        text += strings + ' '
    return text
test_path='E:\\level4\\second\\NLP\\project\\deep-spell-checkr-master\\data'
test_book=['input.txt']
test_sentence = read_text(test_path, test_book)
true_path='E:\\level4\\second\\NLP\\project\\deep-spell-checkr-master\\data'
true_book=['true.txt']
true_sentences = read_text(true_path, true_book)
string=''
arr=[]
for x in true_sentences:
    if(x==' '):
        arr.append(string)
        string=''
    else:
        string =string+x
true=arr    
if __name__ == '__main__':
    text  = read_text(data_path, books)
    vocab = tokenize(text)
    vocab = list(filter(None, set(vocab)))
    # `maxlen` is the length of the longest word in the vocabulary
    # plus two SOS and EOS characters.
    maxlen = max([len(token) for token in vocab]) + 2
    train_encoder, train_decoder, train_target = transform(
        vocab, maxlen, error_rate=error_rate, shuffle=False)

    tokens = tokenize(test_sentence)
    tokens = list(filter(None, tokens))
    nb_tokens = len(tokens)
    misspelled_tokens, _, target_tokens = transform(
        tokens, maxlen, error_rate=error_rate, shuffle=False)

    input_chars = set(' '.join(train_encoder))
    target_chars = set(' '.join(train_decoder))
    input_ctable = CharacterTable(input_chars)
    target_ctable = CharacterTable(target_chars)
    
    encoder_model, decoder_model = restore_model(model_path, hidden_size)
    
    input_tokens, target_tokens, decoded_tokens = decode_sequences(
        misspelled_tokens, target_tokens, input_ctable, target_ctable,
        maxlen, reverse, encoder_model, decoder_model, nb_tokens,
        sample_mode=sample_mode, random=False)
    
    print('-')
    print('Input sentence:  ', ' '.join([token for token in input_tokens]))
    print('-')
    print('Decoded sentence:', ' '.join([token for token in decoded_tokens]))
    print('-')
    print('Target sentence: ', ' '.join([token for token in target_tokens]))
    
    
   
    
decoded_tokens[1]
#the four

#VAL_MAXLEN = 16

  
def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i]==y_pred[i]:
           TP += 1
        if y_pred[i] and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]:
           TN += 1
        else:
            if y_actual[i]!=y_pred[i]:
               FN += 1

    return(TP, FP, TN, FN)

TP, FP, TN, FN=perf_measure(true, decoded_tokens)
accuracy=(TP+TN)/(TP+TN+FP+FN)
recall=TP/(TP+FN)
precision=TP/(TP+FN)
Fscore=2*((precision*recall)/(precision+recall))
