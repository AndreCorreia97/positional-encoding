# Implementation of Positional Encoding to Control Output Sequence Length

- Implementation of the following [paper](https://www.aclweb.org/anthology/N19-1401) by Sho Takase, Naoaki Okazaki

- Adapted from their [implementation](https://github.com/takase/control-length)

## Setup

> Clone the repository
- git clone https://github.com/AndreCorreia97/positional-encoding

> Install the requirements
- pip install -r requirements.txt

> Download and extract dataset or use your own
- [link](https://drive.google.com/file/d/1L4-wI2HUqLXgblRTDNn42V75Vvof6esN/view?usp=sharing) to the adapted gigaWord dataset

> Install fairseq requirements by running setup.py
- python setup.py install

## Pre-Process the data
- python preprocess.py --source-lang wp_source --target-lang wp_target --trainpref gigaWord/train --validpref gigaWord/valid --testpref gigaWord/test --joined-dictionary  --destdir data-bin/gigaWord        

## (Alternatively) Pre-process only the test data (for summarizing specific input file)
> Place the summarization's goal file in gigaWord/test
- python preprocess.py --source-lang wp_source --target-lang wp_target  --testpref gigaWord/test --joined-dictionary  --destdir data-bin/gigaWord

## Train the model
> If program runs out of memory, reduce the token size or max sentences with --max_sentences, increase epoch to compensate
- python train.py data-bin/gigaWord --source-lang wp_source --target-lang wp_target --arch transformer_wmt_en_de --optimizer adam  --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07  --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 3584 --seed 2723 --max-epoch 100 --update-freq 16 --share-all-embeddings --represent-length-by-lrpe --ordinary-sinpos --save-dir output

> LRPE version: --represent-length-by-lrpe

> LDPE version: --represent-length-by-ldpe

> PE version: --ordinary-sinpos

> Trained Model with PE:

- [link](https://drive.google.com/file/d/1AKz9yxh7DA4cDzQgtMwhKeinEhWNoEdm/view?usp=sharing)

## Summarize Test file
> Change the desired length parameter to the size you desire
- python summarize.py data-bin/gigaWord --source-lang wp_source --target-lang wp_target --path output/checkpoint_best.pt --desired-length 75 --batch-size 32 --beam 5

## Acknowledgements

> Fairseq, as used by the authors adapted from: [repository](https://github.com/pytorch/fairseq).
