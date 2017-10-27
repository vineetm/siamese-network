# siamese-network
Experiments with text similarity using Siamese network

We will try to reproduce Sec 4.2 and 4.3 of the paper [Training End-to-End Dialogue Systems with the Ubuntu Dialogue Corpus](https://www.google.co.in/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwiJ79Ggk_fWAhXLMI8KHQmPDaIQFggnMAA&url=http%3A%2F%2Fdad.uni-bielefeld.de%2Findex.php%2Fdad%2Farticle%2Fdownload%2F3698%2F3593&usg=AOvVaw1NmiKknJz-6RXw5cAe-Sop)

We will implement our model in Tensorflow `tf1.3` and also use tensorflow dataset API for working with data `tf.contrib.data`

#### High-Level Goal

##### Retrieval based model
* Assign a score to a text pair (context, utterance)
* Use scorer to select utterance from a list of candidates
* Measure performance by selecting from 10 candidates:
  * R@1: Does top candidate equal GT
  * R@2: Does any of Top-2 candidates equal GT
  * R@5: Does any of Top-5 candidates equal GT

##### Dual Encoder
* Use a RNN to compute embedding of context $c$
* Use a RNN to compute embedding of utterance $r$
* Measure similarity between $c$ and $r$
  * score = $\sigma(c^TMr)$


#### Data Prep
  * Download and prepare [raw data](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
    ```bash
    git clone git@github.com:rkadlec/ubuntu-ranking-dataset-creator.git

    cd ubuntu-ranking-dataset-creator/src

    # Still need to verify if this was the approach used
    ./generate.sh -t

    ## ./generate.sh -t -s -l
    ```

  * This will create `train.csv, valid.csv, test.csv`
    ```bash
    wc -l *.csv
    18921 test.csv
    1000001 train.csv
    19561 valid.csv
    ```

  * Prep data for train and validation
    * For training: `train.txt1` `train.txt2` `train.labels`

    * Validation: Original valid data has 10 parts (GT, 9 distractors). We would create a positive

      * `valid.txt1`, `valid.txt2`, `valid.labels`

    * For Retrieval metric: `valid.txt1` `valid.txt2.p0` ... `valid.txt2.p9`.
      * p0 is Ground Truth (GT)

    ```python
      python prep_data.py $RAW_DATA_DIR $OUT_DIR
    ```
    See `prep_data.sh` for sample parameters

#### Experiments
1. Opt='sgd', lr=0.01, 0.1, 1.0
  * V = 100K, Word embedding matrix initialized using truncated_normal
  * Batch_size: 256
  * M = initialized using truncated_normal
