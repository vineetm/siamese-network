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
