#!/bin/bash

module purge 
ml easybuild  icc/2017.1.132-GCC-6.3.0-2.27  impi/2017.1.132 
ml python/3.6.0
ml r/3.5.1
ml MultiQC/1.3-Python-3.6.1
ml fastqc/0.11.5
#Loads all modules 

r1='/home/bcosgrov/bgmp/fungroup/testFiles/testR1.fq.gz' # Location of read 1 file
r2='/home/bcosgrov/bgmp/fungroup/testFiles/testR4.fq.gz' # location of read 2 file
i1='/home/bcosgrov/bgmp/fungroup/testFiles/testR2.fq.gz' # location of index 1 file
i2='/home/bcosgrov/bgmp/fungroup/testFiles/testR3.fq.gz' # location of index 2 file 
map='/home/bcosgrov/bgmp/fungroup/AMF_library-mappingfile.csv' # location of experiment mapping file 
p1='CAGCCGCGGTAATTCCAGCT' #Primer 1 sequence, ie. WANDA
p2='GAACCCAAACACTTTGGTTTCC' #Primer 2 Sequence ie. AML2
q=5 # Minimum Line Quality score to be demultiplexed 
t=2 # Tolerance of sequence to base pair mismatches in primer1 and primer 2 


/usr/bin/time -v ./orient.py -R $r1 -r $r2 -I $i1 -i $i2 -b $map -t $t -P $p1 -p $p2 -q $q 

mkdir samples 
mv *.fastq samples
cd samples
mkdir reports 
fastqc * -o reports
cd reports
multiqc .

for f in *_*; do mv -v "$f" $(echo "$f" | tr '_' '-'); done
rename -- -R2.fastq.gz _R2.fastq.gz *R2

rename -- -R1.fastq.gz _R1.fastq.gz *R1
