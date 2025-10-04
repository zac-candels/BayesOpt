import subprocess 
import os
import re
import time
import shutil

import warnings
warnings.filterwarnings("ignore")

def modSlurmFile(newInputName: str):

    shutil.copy('../LBM2D/LB_sim/input.txt', newInputName)


def main():
    newInputName = "../LBM2D/LB_sim/input_x0.34_y2.1.txt"
    modSlurmFile(newInputName)

 
main()