import pandas as pd
# Load  data from .txt files
attributes_df= pd.read_csv('data/list_attr_celeba.txt', delim_whitespace=True, header=1)
landmarks_df = pd.read_csv('data/list_landmarks_align_celeba.txt', delim_whitespace=True, header=1)
landmarks_df = landmarks_df.astype('float32')

# Identities has no header, so we need to specify it, use first column as index and rename the second column to 'identity'
identities_df = pd.read_csv('data/identity_CelebA.txt', delim_whitespace=True, header=None, index_col=0, names=['identity'])

# Combine the two dataframes
combined_df = pd.concat([attributes_df, landmarks_df, identities_df], axis=1)

# Sort the dataframe by identity
combined_df.sort_values('identity', inplace=True)

split_location = combined_df.iloc[-1]['identity'] // 2

# Split the dataframe into two
private_df = combined_df[combined_df['identity'] <= split_location]
public_df = combined_df[combined_df['identity'] > split_location]

private_df.to_pickle('data/private_df.pkl')
public_df.to_pickle('data/public_df.pkl')

# TODO: We need to also add augmentation to the data