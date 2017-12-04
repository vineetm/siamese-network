We work with [Ubuntu Dialogue Corpus v2](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
  * We use the latest repository (Last Commit: 18 Oct 2017)
  * We only `tokenize` the text.
  * We further create a larger retrieval dataset with (50,)
  * Detailed Steps:
    ```
    git clone git@github.com:rkadlec/ubuntu-ranking-dataset-creator.git

    cd ubuntu-ranking-dataset-creator/src
    ```

    ```bash
    #!/usr/bin/env bash

    # download punkt first
    python download_punkt.py

    python create_ubuntu_dataset.py -t --output 'train.csv' 'train'

    for n in 9 49 99 499 999 4999 9999 19999 49999; do
    python create_ubuntu_dataset.py -t --output "test.${n}.csv" 'test' -n $n
    echo "$k test done"

    python create_ubuntu_dataset.py -t --output "valid.${n}.csv" 'valid' -n $n
    echo "$k valid done"
    done
    ```
