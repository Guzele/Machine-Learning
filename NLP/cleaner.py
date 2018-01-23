import os
import re

import general as g #import data_directory, source_directory, cleaned_directory, samples_directory , train_test_directory
#from alphabet import languages_dict

# remove new lines - we need dense data
def remove_newlines(text):
    return text.replace('\n', ' ') 

def remove_newlines(text):
    return text.replace('\n', ' ') 

def remove_dashes(text):
    return text.replace('– ', '') 
    
# replace many spaces in text with one space - too many spaces is unnecesary
# we want to keep single spaces between words
# as this can tell DNN about average length of the word and this may be useful feature
def remove_manyspaces(text):
    return re.sub(' +',' ', text)

def clean_text(text):
    #text = remove_newlines(text)
    text = remove_manyspaces(text)
    text = remove_dashes(text)
    return text


def main():
    for lang_code in g.languages_dict:
        path_src = os.path.join(g.source_dir, lang_code+".txt")
        f = open(path_src)
        content = f.read()
        print('Language : ',lang_code)
        print ('Content before cleaning :-> ',content[1000:1000+g.text_sample_size])
        f.close()
        # cleaning
        content = clean_text(content)
        print ('Content after cleaning :-> ',content[1000:1000+g.text_sample_size])
        path_cl = os.path.join(g.cleaned_dir,lang_code + '.txt')
        f = open(path_cl,'w')
        f.write(content)
        f.close()
        del content
        print ("Cleaning completed for : " + path_src,'->',path_cl)
        print (100*'-')
    print ("END OF CLEANING")

if __name__ == "__main__":
    g.init()
    main()


