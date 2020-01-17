#!/usr/bin/env bash
#!/bin/sh

sbatch --job-name=reddit_bert_pretraining_tf2 --output=reddit_bert_pretraining_tf2.out ./reddit/submit_scripts/emayhem/run_pretraining.sh

