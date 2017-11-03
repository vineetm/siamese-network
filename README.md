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
* Parameters
  * Opt: 'Adam'
  * Batch_Size: 256
  * M: eye
  * Word embeddings: init truncated_normal
  * d: 128

  * lr = 0.0005, 0.0065, 0.0075, 0.001
  * dr = 0.0, 0.1, 0.2, 0.4, 0.6, 0.8

  * Results
    * dr=0.0
      * lr=0.001, Valid_Loss: 0.5455, Step: 6K, Train_Loss: 0.4903
      * R@1:0.4784 R@2: 0.6638 R@5:0.9035
    * dr=0.1
      * lr=0.001, Valid_Loss: 0.5381, Step: 10K, Train_Loss: 0.4368
      * R@1:0.5131 R@2: 0.6949 R@5:0.9206
    * dr=0.2
      * lr=0.0005, Valid_Loss: 0.5268, Step: 27K, Train_Loss: 0.4052
      * R@1:0.5486 R@2: 0.7276 R@5:0.9362
    * dr=0.2,
      * Valid_Loss: 0.5332 Train_Loss:0.41 Step: 17K
      * lr=0.001, R@1:0.5445 R@2: 0.7233 R@5:0.9325
    * dr=0.4
      * Valid_Loss: 0.5156, Train: 0.4264, Step: 44K, lr=0.00065
      * R@1:0.5672 R@2: 0.7448 R@5:0.9441
    * dr=0.6
      * lr=0.005, Valid_Loss:0.5189, Step:125K
      * **R@1:0.5847 R@2: 0.7605 R@5:0.9482**
