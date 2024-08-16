#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=autoencoder_$1
#SBATCH --output=development/test_ae/autoencoder/$1/slurm_%a_%j.out
#SBATCH --error=development/test_ae/autoencoder/$1/slurm_%a_%j.err
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=clara
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB


ml cuDNN
# set for cuda reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096
# set for modin parallelization
export __MODIN_AUTOIMPORT_PANDAS_=:1

cd development/test_ae/autoencoder
source venv-gallia/bin/activate
make prediction_only  RUN_ID=$1
exit 0
EOT
