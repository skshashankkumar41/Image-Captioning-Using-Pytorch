import pandas as pd
import re 
import numpy as np 
import string 
from string import digits

def split_df(df_path, save_path, train_percent=.8, validate_percent=.1, seed=None):
    df = pd.read_csv(df_path)
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    
    test = df.iloc[perm[validate_end:]]
    
    train.to_csv(save_path + 'train_df.csv',index = False)
    test.to_csv(save_path + 'test_df.csv',index = False)
    validate.to_csv(save_path + 'validate_df.csv',index = False)
    return True

def preprocess(df_path,save_path):
    df = pd.read_csv(df_path)

    df=df.dropna()

    df['caption'] = df['caption'].apply(lambda x: x.lower())

    df['caption']= df['caption'].apply(lambda x: re.sub("'", '', x))

    exclude = set(string.punctuation) # Set of all special characters
    
    # Remove all the special characters
    df['caption'] = df['caption'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

    remove_digits = str.maketrans('', '', digits)
    df['caption'] = df['caption'].apply(lambda x: x.translate(remove_digits))


    # Remove extra spaces
    df['caption'] = df['caption'].apply(lambda x: x.strip())
    df['caption'] = df['caption'].apply(lambda x: re.sub(" +", " ", x))
    
    df.to_csv(save_path + 'preprcessedData.csv',index = False)
    #df = df.iloc[:,1:]

    splitted = split_df(save_path + 'preprcessedData.csv', save_path)

    if splitted:
        return "Splitting Done! Saved"
    else:
        return "Error while Splitting"

if __name__ == "__main__":
    preprocess('input/captions.txt', 'input/')


