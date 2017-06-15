#!/bin/bash
#SBATCH --mem=10000MB

module load python3
~/myapp/bin/python3.5 xgb1.py