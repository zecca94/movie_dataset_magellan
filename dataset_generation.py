#!/usr/bin/env python3

import py_entitymatching as em
import pandas as pd

# Load CSV input files
path_ds1 = 'dataset/dataset1.csv'
path_ds2 = 'dataset/dataset2.csv'
path_gt = 'dataset/gt.csv'
ds1 = em.read_csv_metadata(path_ds1)
ds2 = em.read_csv_metadata(path_ds2)
gt = em.read_csv_metadata(path_gt)

# Fit schemas and drop useless attributes
ds1 = ds1.rename(columns={'starring': 'actors'})
ds2 = ds2.rename(columns={'actor name': 'actors'})
ds1 = ds1.drop(columns=['editor', 'writer'])
ds2 = ds2.drop(columns=['director name', 'genre', 'imdb_ksearch_id', 'url', 'year'])
em.set_key(ds1, 'id')
em.set_key(ds2, 'id')

# Downsampling of ds1 and ds2
# ds1, ds2 = em.down_sample(ds1, ds2, 3000, 1, show_progress=False)

# Create an overlap blocker and do blocking on the attribute 'title'
ob = em.OverlapBlocker()
bt = ob.block_tables(ds1, ds2, 'title', 'title', \
                         l_output_attrs=['id', 'title', 'actors'], \
                         r_output_attrs=['id', 'title', 'actors'], \
                         overlap_size=2, show_progress=False)
print('Number of tuples in blocks: ' + str(len(bt)))

# Detect the matching couples in the block (join with ground truth)
matches = pd.merge(bt, gt, left_on=['ltable_id', 'rtable_id'], right_on=['id1', 'id2'])
matches = matches.drop(columns=['id1', 'id2'])
print('Number of matches: ' + str(len(matches)))

# Do the difference to obtain not matching couples
no_matches = pd.concat([bt, matches]).drop_duplicates(keep=False)
print('Number of no_matches: ' + str(len(no_matches)))

# Insert label column
matches['label'] = 1
no_matches['label'] = 0

# Create the dataset (7K matches, 21K no_matches)
sample_matches = matches.sample(7000)
sample_no_matches = no_matches.sample(21000)
dataset = pd.concat([sample_matches, sample_no_matches])
dataset = dataset.rename(columns={'_id': 'id', 'ltable_id': 'left_id', 'rtable_id': 'right_id', \
                                  'ltable_title': 'left_title', 'rtable_title': 'right_title', \
                                  'ltable_actors': 'left_actors', 'rtable_actors': 'right_actors'})
dataset.to_csv('dataset/dataset.csv', index=False)
print('Number of elements in the dataset: ' + str(len(dataset)))

# Create training, validation and test sets (3:1:1)
train_matches = dataset.head(4200)
dataset = pd.concat([dataset, train_matches]).drop_duplicates(keep=False)
valid_matches = dataset.head(1400)
dataset = pd.concat([dataset, valid_matches]).drop_duplicates(keep=False)
test_matches = dataset.head(1400)
dataset = pd.concat([dataset, test_matches]).drop_duplicates(keep=False)
train_no_matches = dataset.head(12600)
dataset = pd.concat([dataset, train_no_matches]).drop_duplicates(keep=False)
valid_no_matches = dataset.head(4200)
dataset = pd.concat([dataset, valid_no_matches]).drop_duplicates(keep=False)
test_no_matches = dataset.head(4200)
dataset = pd.concat([dataset, test_no_matches]).drop_duplicates(keep=False)

train = pd.concat([train_matches, train_no_matches])
valid = pd.concat([valid_matches, valid_no_matches])
test = pd.concat([test_matches, test_no_matches])

train = train.sample(frac=1)
valid = valid.sample(frac=1)
test = test.sample(frac=1)

print('Number of elements in the training set: ' + str(len(train)))
print('Number of elements in the validation set: ' + str(len(valid)))
print('Number of elements in the test set: ' + str(len(test)))

train.to_csv('dataset/train.csv', index=False)
valid.to_csv('dataset/valid.csv', index=False)
test.to_csv('dataset/test.csv', index=False)
