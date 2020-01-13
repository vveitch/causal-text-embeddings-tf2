#!/bin/bash

# Each Peer read dataset_ is pre-divided into train/dev/test. Merge these into "all"

#PeerDir=/home/victor/Documents/causal-spe-embeddings/dat/PeerRead
PeerDir=/home/victor/Documents/causal-spe-embeddings/dat/PeerRead/nips_2013-2017

for dir in $PeerDir*/; do
    for subdir in $dir*/; do
	echo $subdir;
	cp -RT $subdir/ $dir/all/
    done
done
