chmod +x $0

datasets=("synthetic")
models=("adone")
# outlier_types=('contextual')
flip_rates=(0.0 0.5 1.0)

# Define the output directory for log files
output_dir="output"
mkdir -p $output_dir

# Loop through datasets, models, outlier types, and flip rates

for flip_rate in "${flip_rates[@]}"; do
  output_file="$output_dir/model_guide_flip_${flip_rate}_nocf.log"
  command="python rw_tune_model.py --flip_rate $flip_rate >> $output_file 2>&1 &wait"
  echo $command
  eval $command
done


# Indicate successful execution of all commands
echo "All commands executed successfully." 
