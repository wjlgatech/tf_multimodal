# tf-multimodal

## Objectives

To build a automated Tensorflow based multimodal classifier that consumes tabular data with cnt, cat, txt, img columns.

## Notes
- this package is developed using [nbdev_colab](https://github.com/muellerzr/nbdev_colab) 
- tf_multimodal is a synergetic project [auto-tfrs](https://github.com/wjlgatech/auto_tfrs), an easy-to-use, easy-to-make recommendation engineer based on [tfrs](https://www.tensorflow.org/recommenders)

## Milestones
- [5/5] build a working [notebook](https://github.com/wjlgatech/tf-multimodal/blob/main/tf_multimodal.ipynb) with 1 sample dataset
- [/5] test performance on other datasets
- [/5] modularize it and put in .py
- [/5] packaging it into a library

## Features

**Features Built**
- [5/5] preprocess and encode cnt_cols: normalization and bucketization

- [5/5] preprocess and encode cat_cols for int_cat_cols and str_cat_cols

- [5/5] preprocess and encode txt_cols: implemented 2 methods: 1.LSTM 2.Bert

- [5/5] preprocess and encode img_cols
  - txt+img multimodal: 
    - https://keras.io/examples/nlp/multimodal_entailment/

- [5/5]**Situation**: How do I combine info from various dtype of data, given that each dtype have different number of columns (e.g. 5 cnt_cols, 10 cat_cols, 2 txt_cols, 2 img_cols),  and the embedding of each dtype column can be very different emb_width (e.g. con_cols have emb_width 1 vs. cat_col has emb_width 16). The info from col with narrow emb_width can be overwhelmed by the col with wide emb_width. **Solution**: instead of using simple concat, to add a `deep-wide module` where the deep layer squeeze the cols with dense representation (e.g. embedding with 16 cols) while the wide layer is to process the sparse cols. Then the output from deep-branch and the output from the wide-branch is combined. In this way, features of various embedding width are normalized so that their contribution can be backtracked and compared.
  - keyref1: https://keras.io/examples/structured_data/wide_deep_cross_networks/
  - keyref2: https://branyang.gitbooks.io/tfdocs/content/tutorials/wide_and_deep.html



  - https://keras.io/examples/nlp/text_classification_with_transformer/



- [5/5] automate the separation for cnt, cat, txt columns in df

- [5/5] OneTower structure: concat the emb of txt_cols, cnt_cols, cat_cols for df

- [5/5] allow txt_, cnt_, cat_cols interaction with deep cross module

- [5/5] to accelerate hyperparameter tuning by a tf.keras-based module https://keras.io/keras_tuner/, build 3 tuners: randomSearch, Bayesian, Hyperband.

- [5/5] in order to experiment various downstream learning architectures (e.g. simple, deep-wide-cross, dcn, two tower), simplify and modularize the preprocessing and encoding of cnt, cat, txt, img.

- [5/5] to automatically switch between binary classification and multiclass classification, build unified deep learning utilities, including data-preprocessing for label column (multi-hot-encoding), loss function (tf.keras.losses.CategoricalCrossentropy) and custom metrics (accuracy, F1, roc-auc, pr-auc, confusion matrix, avg_precision, avg_recall)

- [5/5] wide, deep, cross on concat_emb
https://keras.io/examples/structured_data/wide_deep_cross_networks/
  - cat dense emb VS cat sparse emb
  
**Features to Build**

- [/5] to do tf-model-specific Feature selection, use grn and vsn: https://keras.io/examples/structured_data/classification_with_grn_and_vsn/

- [/5] trace feature contribution on both population-level and individual-sample-level, use Shap with col_emb_ls

- [/5] to do tf-model-specific Feature Selection, experiment adding a attention layer
  - https://towardsdatascience.com/create-your-own-custom-attention-layer-understand-all-flavours-2201b5e8be9e

  - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention

  - https://analyticsindiamag.com/a-beginners-guide-to-using-attention-layer-in-neural-networks/

- [/5] To build a two-tower structure (one for User, another for Item) for a hybrid recommendation system, concat the emb of txt_cols, cnt_cols, cat_cols for user and item, respectively.

- [/5] Instead of making point-prediction, make interval-prediction with a confidential interval. https://mapie.readthedocs.io/en/latest/index.html

**References & Credits**:

- tf cnt+cat: https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers

- cat + img: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

- cnt + txt: https://towardsdatascience.com/combining-numerical-and-text-features-in-deep-neural-networks-e91f0237eea4

- txt bert https://www.tensorflow.org/text/tutorials/classify_text_with_bert

