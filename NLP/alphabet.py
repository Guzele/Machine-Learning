import os
import general as g

# utility function to turn language id into language code
def decode_langid(langid):    
    for dname, did in g.languages_dict.items():
        if did == langid:
            return dname

# we will use alphabet for text cleaning and letter counting
def define_alphabet():
    base_en = 'abcdefghijklmnopqrstuvwxyz'

    #??????????????????????????????????
    special_chars = ' !?¿¡-’„”“”'

    czech = 'áčďéěíjňóřšťúůýž'
    polish = 'ąćęłńóśźż'
    slovak = 'áäčďdzdžéíĺľňóôŕšťúýž'
    hungarian = 'áéíöóőüúű'

    all_lang_chars = base_en + czech + polish + slovak
    small_chars = list(set(list(all_lang_chars)))
    small_chars.sort() 
    big_chars = list(set(list(all_lang_chars.upper())))
    big_chars.sort()

    #???????????????????????????????????
    small_chars += special_chars
    letters_string = ''
    letters = small_chars #+ big_chars
    for letter in letters:
        letters_string += letter
    return letters # big_chars,small_chars, 

def main():
    alphabet = define_alphabet()
    print (alphabet)

if __name__ == "__main__":
    g.init()
    main()
