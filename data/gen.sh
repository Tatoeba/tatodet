#!/bin/bash

#wget https://downloads.tatoeba.org/exports/sentences.tar.bz2
#tar -jxvf sentences.tar.bz2

n=1000
l=('eng' 'fra' 'por' 'spa' 'ita' 'glg' 'cat' 'jpn' 'cmn' 'kor');

for i in ${l[@]};
   do grep -P "\t""$i" sentences.csv | shuf -n $n > "$i""_$n.csv"
 done

n=200
echo -ne > "testset.csv"
for i in ${l[@]};
  do grep -P "\t""$i" sentences.csv | shuf -n $n >> "testset.csv"
done
