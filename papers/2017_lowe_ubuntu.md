**Paper**: Training End-to-End Dialogue Systems with the Ubuntu Dialogue Corpus
Lowe et. al, 2017

Dataset github link: `https://github.com/rkadlec/ubuntu-ranking-dataset-creator`

Pre-trained word vectors: `http://nlp.stanford.edu/data/glove.840B.300d.zip`

### Data-prep
* Clone the `github dataset repo`
  ```bash
  git clone git@github.com:rkadlec/ubuntu-ranking-dataset-creator.git
  ```

* Prepare train, valid and test data

  ```bash
  cd ubuntu-ranking-dataset-creator/src
  ./generate.sh -t -s -l
  ```

* Separate out `ctx, utterance and label` and build vocabulary

### Dataset Description
1. Training Data:
context, utterance, label

2. Valid/Test Data: context, utterance, distractors
    * distractors are sampled randomly from test set.
