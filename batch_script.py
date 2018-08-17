"""

Stein_GAN: batch_script.py

Created on 8/6/18 12:30 PM

@author: Hanxi Sun

"""

import os

EXP = "plots_2d"
SCRIPT_DIR = os.getcwd() + "/job_script/" + EXP + "/"
SERVER_DIR = "/home/sun652/Stein_GAN/"
if not os.path.exists(SCRIPT_DIR):
    os.makedirs(SCRIPT_DIR)

PYTHON_FILE = SERVER_DIR + "scr/" + EXP + "_server-2.py"

z_dim_lst = [5, 10, 20]
n_d_lst = [5, 10, 20]
md_lst = [4, 4.5, 5, 6]
lr_lst = [0.001, 0.0001, 0.00001]
N_rep = 5

print(len(z_dim_lst) * len(n_d_lst) * len(md_lst) * len(lr_lst) * N_rep)

############################################################
#                       Job Scripts                        #
############################################################

JOB_PREFIX = "JOB"
JOB_DIR = SERVER_DIR + "jobs/" + EXP + "/"
BATCH_SUBMIT = SCRIPT_DIR + "batch_submit.sh"
WALLTIME = "02:30:00"

PBS = ("#PBS -V \n" +
       "#PBS -q standby \n" +
       "#PBS -l walltime=" + WALLTIME + " \n\n")

submit = open(BATCH_SUBMIT, 'w')

submit.write("#!/bin/sh\n\n")
# submit.write("module load anaconda/5.1.0-py36\n\n")

idx = 0
for z_dim in z_dim_lst:
    for n_d in n_d_lst:
        for md in md_lst:
            for lr in lr_lst:
                for _ in range(N_rep):
                    job_name = JOB_PREFIX + "_{}".format(idx)
                    job = open(SCRIPT_DIR + job_name + ".txt", 'w')
                    job.write("#!/bin/sh -l\n" +
                              "# FILENAME: " + job_name + "\n\n" +
                              PBS +
                              "python " + PYTHON_FILE + " {0} {1} {2} {3} \n\n".format(z_dim, n_d, md, lr))
                    job.close()
                    idx += 1

                    submit.write("qsub " + JOB_DIR + job_name + ".txt \n")
submit.close()


############################################################
#                       Delete Jobs                        #
############################################################
id0, id1 = 7715376, 7715772

for i in range(id0, id1+1):
    print("qdel {}".format(i))










