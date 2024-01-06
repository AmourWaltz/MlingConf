# Copyright 2023        Author: XUE Boyang
# Afflition: MoE Key Lab, The Chinese University of Hong Kong.


##################################################################################################
# Display current time.
##################################################################################################
echo $(date +%Y-%m-%d" "%H:%M:%S)

echo "$0 $*"  # Print the command line for logging

stage=$1


##################################################################################################
# Stage 1: LLM question answering part.
##################################################################################################

if [ $stage == infer ]; then
  echo "$0: LLM question answering part."
    python code/$stage.py
fi

##################################################################################################
# Stage 2: LLM evaluation part on several QA metrics.
##################################################################################################

if [ $stage == eval ]; then
  echo "$0: LLM evaluation part on several QA metrics."
    python code/$stage.py
fi
