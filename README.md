# bgmp-group-project-vineyard_fungal_assessment
Pipeline for demultiplexing, deduplication, DADA2, & variance transformation of amplicon sequencing data
generated from a novel adapter scheme developed in the McGuire Lab (University of Oregon)


All dependancies required to run this pipeline exist within enviroment XG26.yml 

prior to running this pipeline please install the enviroment from the.yml file located in the env folder

$conda env create --name XGboost --file=XG26.yml
$conda activate XGboost

All relavent .sh and .py files required for our pipeline are located within the scripts folder

Analysis and results generated on TALAPAS

NOTE : While running on Talapas the given .sh scripts need to be given #sbatch parameters and the first script requires to be executed in the same folder as the sam files you wish to run analysis on.

![](/figures/pipeline_diagram.png)

ORDER: orient.py > align_multiple.R > dedupSAM.sh > dada2pipeline.R

INPUTS: Raw sequencing FASTQ output with barcode mapping file, according to the specifications of the Genomics & Cell Characterization Core Facility (GC3F) at the University of Oregon.

OUTPUTS: ASV_table.tsv : ASV+taxonomy table DADA2 outputs ASV_summary.tsv : Sum counts of all ASV's per sample track_reads.csv : Table counting the number of reads kept at each step of the pipeline ASV-VT.tsv : A variance transformed version of ASV_table.tsv phyloseq_object.RDS : phyloseq object of the variance transformed ASV with metadata
