
sbatch --job-name=uboost --error=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/uboost.err --output=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/uboost.out uboost.sh

sbatch --job-name=ANN_lambda1 --error=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/ANN_lambda1.err --output=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/ANN_lambda1.out ANN_lambda1.sh

sbatch --job-name=ANN_lambda3 --error=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/ANN_lambda3.err --output=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/ANN_lambda3.out ANN_lambda3.sh

sbatch --job-name=ANN_lambda10 --error=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/ANN_lambda10.err --output=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/ANN_lambda10.out ANN_lambda10.sh

sbatch --job-name=disco_lambda1 --error=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/disco_lambda1.err --output=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/disco_lambda1.out disco_lambda1.sh

sbatch --job-name=disco_lambda5 --error=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/disco_lambda5.err --output=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/disco_lambda5.out disco_lambda5.sh

sbatch --job-name=disco_lambda10 --error=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/disco_lambda10.err --output=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/disco_lambda10.out disco_lambda10.sh

sbatch --job-name=disco_lambda30 --error=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/disco_lambda30.err --output=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/disco_lambda30.out disco_lambda30.sh

sbatch --job-name=disco_lambda50 --error=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/disco_lambda50.err --output=/hpcfs/bes/mlgpu/gang/adversarial/jobs/log/disco_lambda50.out disco_lambda50.sh
