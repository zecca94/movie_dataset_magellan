#!/usr/bin/env python3

import py_entitymatching as em
import pandas as pd
import time

# Load CSV input files
path = 'dataset/overlap_title_2/'
path_ds1 = path + 'ds1.csv'
path_ds2 = path + 'ds2.csv'
path_train = path + 'train.csv'
path_valid = path + 'valid.csv'
path_test = path + 'test.csv'

ds1 = em.read_csv_metadata(path_ds1, key='id')
ds2 = em.read_csv_metadata(path_ds2, key='id')
train = em.read_csv_metadata(path_train, key='id', ltable=ds1, rtable=ds2, fk_ltable='left_id', fk_rtable='right_id')
valid = em.read_csv_metadata(path_valid, key='id', ltable=ds1, rtable=ds2, fk_ltable='left_id', fk_rtable='right_id')
test = em.read_csv_metadata(path_test, key='id', ltable=ds1, rtable=ds2, fk_ltable='left_id', fk_rtable='right_id')

# Process the original datasets to make the attributes 'actors' of the same type ('str_gt_10w')
ds1['actors'] = ds1['actors'].apply(lambda x: str(x) + ' ' + str(x))

# Generate a set of features for matching
match_t = em.get_tokenizers_for_matching()
match_s = em.get_sim_funs_for_matching()
atypes1 = em.get_attr_types(ds1)
atypes2 = em.get_attr_types(ds2)
match_c = em.get_attr_corres(ds1, ds2)
match_c['corres'].remove(('id', 'id'))

# feature_table = em.get_features_for_matching(ds1, ds2, validate_inferred_attr_types=False)
feature_table = em.get_features(ds1, ds2, atypes1, atypes2, match_c, match_t, match_s)

# Convert the training set into a set of feature vectors using the feature table
feature_vectors = em.extract_feature_vecs(train, feature_table=feature_table, attrs_after='label', \
                                          show_progress=False)
feature_vectors = feature_vectors.fillna(0.0)

# Create a set of ML matchers
decision_tree = em.DTMatcher(name='DT', random_state=0)
random_forest = em.RFMatcher(name='RF', random_state=0)
svm = em.SVMMatcher(name='SVM', random_state=0)
naive_bayes = em.NBMatcher(name='NB')
linear_regression = em.LinRegMatcher(name='LinReg')
logistic_regression = em.LogRegMatcher(name='LogReg', random_state=0)

# Train using feature vectors
decision_tree.fit(table=feature_vectors, exclude_attrs=['id', 'left_id', 'right_id', 'label'], target_attr='label')
random_forest.fit(table=feature_vectors, exclude_attrs=['id', 'left_id', 'right_id', 'label'], target_attr='label')
svm.fit(table=feature_vectors, exclude_attrs=['id', 'left_id', 'right_id', 'label'], target_attr='label')
naive_bayes.fit(table=feature_vectors, exclude_attrs=['id', 'left_id', 'right_id', 'label'], target_attr='label')
linear_regression.fit(table=feature_vectors, exclude_attrs=['id', 'left_id', 'right_id', 'label'], target_attr='label')
logistic_regression.fit(table=feature_vectors, exclude_attrs=['id', 'left_id', 'right_id', 'label'], target_attr='label')

# Convert the testing set into a set of feature vectors using the feature table
test_feat_vecs = em.extract_feature_vecs(test, feature_table=feature_table, attrs_after='label', show_progress=False)
test_feat_vecs = test_feat_vecs.fillna(0.0)

# Do predictions
start_time = time.time()
dt_predictions = decision_tree.predict(table=test_feat_vecs, exclude_attrs=['id', 'left_id', 'right_id', 'label'], \
                                       append=True, target_attr='prediction', inplace=True)
finish_time = time.time()
em.set_property(dt_predictions, 'fk_ltable', 'left_id')
em.set_property(dt_predictions, 'fk_rtable', 'right_id')
eval_result = em.eval_matches(dt_predictions, 'label', 'prediction')
print('DECISION TREE PREDICTIONS')
print('Elapsed time: ' + str(finish_time - start_time) + ' seconds')
em.print_eval_summary(eval_result)

test_feat_vecs = test_feat_vecs.drop(columns=['prediction'])
start_time = time.time()
rf_predictions = random_forest.predict(table=test_feat_vecs, exclude_attrs=['id', 'left_id', 'right_id', 'label'], \
                                       append=True, target_attr='prediction', inplace=True)
finish_time = time.time()
em.set_property(rf_predictions, 'fk_ltable', 'left_id')
em.set_property(rf_predictions, 'fk_rtable', 'right_id')
eval_result = em.eval_matches(rf_predictions, 'label', 'prediction')
print('RANDOM FOREST PREDICTIONS')
print('Elapsed time: ' + str(finish_time - start_time) + ' seconds')
em.print_eval_summary(eval_result)

test_feat_vecs = test_feat_vecs.drop(columns=['prediction'])
start_time = time.time()
svm_predictions = svm.predict(table=test_feat_vecs, exclude_attrs=['id', 'left_id', 'right_id', 'label'], \
                              append=True, target_attr='prediction', inplace=True)
finish_time = time.time()
em.set_property(svm_predictions, 'fk_ltable', 'left_id')
em.set_property(svm_predictions, 'fk_rtable', 'right_id')
eval_result = em.eval_matches(svm_predictions, 'label', 'prediction')
print('SVM PREDICTIONS')
print('Elapsed time: ' + str(finish_time - start_time) + ' seconds')
em.print_eval_summary(eval_result)

test_feat_vecs = test_feat_vecs.drop(columns=['prediction'])
start_time = time.time()
nb_predictions = naive_bayes.predict(table=test_feat_vecs, exclude_attrs=['id', 'left_id', 'right_id', 'label'], \
                                     append=True, target_attr='prediction', inplace=True)
finish_time = time.time()
em.set_property(nb_predictions, 'fk_ltable', 'left_id')
em.set_property(nb_predictions, 'fk_rtable', 'right_id')
eval_result = em.eval_matches(nb_predictions, 'label', 'prediction')
print('NAIVE BAYES PREDICTIONS')
print('Elapsed time: ' + str(finish_time - start_time) + ' seconds')
em.print_eval_summary(eval_result)

test_feat_vecs = test_feat_vecs.drop(columns=['prediction'])
start_time = time.time()
linreg_predictions = linear_regression.predict(table=test_feat_vecs, \
                                               exclude_attrs=['id', 'left_id', 'right_id', 'label'], append=True, \
                                               target_attr='prediction', inplace=True)
finish_time = time.time()
em.set_property(linreg_predictions, 'fk_ltable', 'left_id')
em.set_property(linreg_predictions, 'fk_rtable', 'right_id')
eval_result = em.eval_matches(linreg_predictions, 'label', 'prediction')
print('LINEAR REGRESSION PREDICTIONS')
print('Elapsed time: ' + str(finish_time - start_time) + ' seconds')
em.print_eval_summary(eval_result)

test_feat_vecs = test_feat_vecs.drop(columns=['prediction'])
start_time = time.time()
logreg_predictions = logistic_regression.predict(table=test_feat_vecs, \
                                                 exclude_attrs=['id', 'left_id', 'right_id', 'label'], append=True, \
                                                 target_attr='prediction', inplace=True)
finish_time = time.time()
em.set_property(logreg_predictions, 'fk_ltable', 'left_id')
em.set_property(logreg_predictions, 'fk_rtable', 'right_id')
eval_result = em.eval_matches(logreg_predictions, 'label', 'prediction')
print('LOGISTIC REGRESSION PREDICTIONS')
print('Elapsed time: ' + str(finish_time - start_time) + ' seconds')
em.print_eval_summary(eval_result)
