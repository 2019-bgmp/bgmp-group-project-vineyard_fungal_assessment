# bgmp-group-project-vineyard_fungal_assessment

All dependancies required to run this pipeline exist within enviroment {NAME}

prior to running this pipeline please install the enviroment from the {NAME}.yml file

$conda env create --name {NAME} --file={NAME}.yml
$conda activate {NAME}

All relavent .sh and .py files required for our pipeline are located within the scripts folder

Analysis and results generated on TALAPAS

(need to figure out how to make input file locations general for our sbatch script)

$sbatch VMP_Fungi_Pipeline.srun
