
def init():
    global data_dir
    data_dir = "./data/"

    global source_dir 
    source_dir = data_dir + 'source'

    global cleaned_dir
    cleaned_dir = data_dir + 'cleaned'
     
    global samples_dir
    samples_dir = data_dir + 'samples'

    global train_test_dir 
    train_test_dir = data_dir + 'train_test'


    # dictionary of languages that our classifier will cover
    global languages_dict
    languages_dict = {'cs':0,'pl':1, 'hu' : 2, 'sk' : 3}
    # length of cleaned text used for training and prediction - 140 chars
    
    global text_sample_size 
    text_sample_size = 140
    # number of language samples per language that we will extract from source files
    global num_lang_samples
    num_lang_samples = 250000
