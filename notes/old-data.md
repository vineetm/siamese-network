* Dataset: Ubuntu Dialogue Corpus v2
* Paper:
  * `Training End-to-End Dialogue Systems with the Ubuntu Dialogue Corpus` [Link](https://www.google.co.in/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwjq6dzn_bXXAhUFS48KHZHUCzYQFggnMAA&url=http%3A%2F%2Fwww.cs.toronto.edu%2F~lcharlin%2Fpapers%2Fubuntu_dialogue_dd17.pdf&usg=AOvVaw3yTYIqpoxwiQSVpEwvHye4)
  * [Github](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)

* This [repo](https://github.com/brmson/dataset-sts/tree/master/data/anssel/ubuntu) has generously released the step-by-step instructions for pre-processing the data.
  ```bash
    git clone git@github.com:rkadlec/ubuntu-ranking-dataset-creator.git
  ```

  ```
  ./generate.sh
  cd ../../

  git clone https://github.com/brmson/tweetmotif
  touch tweetmotif/__init__.py
  ./preprocess.py ubuntu-ranking-dataset-creator/src/train.csv v2-trainset.csv train
  ./preprocess.py ubuntu-ranking-dataset-creator/src/valid.csv v2-valset.csv test
  ./preprocess.py ubuntu-ranking-dataset-creator/src/test.csv v2-testset.csv test
  ```

* Alternatively you can directly download the tokenized text
`v2-trainset.csv`, ...
  ```
  wget http://rover.ms.mff.cuni.cz/~pasky/ubuntu-dialog/v2-trainset.csv.gz
  wget http://rover.ms.mff.cuni.cz/~pasky/ubuntu-dialog/v2-valset.csv.gz
  wget http://rover.ms.mff.cuni.cz/~pasky/ubuntu-dialog/v2-testset.csv.gz
  gunzip v2*.gz
  ```

* Notes:
  * It seems `",` is not tokenized properly. Replace `",` -> `" ,`

* We separate out txt1, txt2 and labels on this csv. Further, to train dual encoder model, we make ratio of pos-negative labels similar to train...

  ```
  python prep_data.py $CSV_DIR -data_dir data
  ```

This creates following data:
  * `train` (`txt1, txt2, labels`): Train data for Dual Encoder. #of positive and negative examples are similar (not exactly same though...)

  * `all.valid`: Validation data with 1 GT and 9 distractors. Used for R@k metrics..

  * `valid`: Validation data with equal pos/neg data points. Used as validation set for dual encoder Networks

  * `pvalid`: Positive validation examples

  * `all.test`: Test data 1 GT, 9 distractors

  * `all.vocab.txt`: Complete vocabulary built using training data (`v2-trainset.csv`) + `UNK`
    * Vocab size: 783, 857
