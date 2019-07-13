#!/bin/bash

function evaluate_unknown_classes {
   arr=( "$@" )

   for v in "${arr[@]}"
   do
      echo "Generating for threshold $v"
      python3 src/utils/class_exclusion.py --predictions_path="predictions_rej_thresh_$v.csv" --results_dir="class_exclusion_$v" --excluded_classes 2 7
      echo "Evaluating"
      python3 src/evaluate.py -p "class_exclusion_$v/predictions.csv" -d "class_exclusion_$v"
   done
}

array=("0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" "0.45" "0.5" "0.55" "0.6" "0.65" "0.7" "0.75" "0.8" "0.8" "0.9" "0.95")
evaluate_unknown_classes "${array[@]}"
