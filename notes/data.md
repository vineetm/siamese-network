We work with [Ubuntu Dialogue Corpus v2](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
  * We use the latest repository (Last Commit: 18 Oct 2017)
  * We only `tokenize` the text.
  * Detailed Steps:

    * Clone repository
      ```
      git clone git@github.com:rkadlec/ubuntu-ranking-dataset-creator.git
      ```

    * Install dependencies. We prefer creating a conda environment, but you can select your favorite method
      ```bash
      conda create -n ubuntu python=2.7
      pip install -r requirements.text
      cd ubuntu-ranking-dataset-creator/src
      ```

   * Finally, generate data. We only `tokenize the text`

      ```bash
      ./generate.sh -t
      ```
