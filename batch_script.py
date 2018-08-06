"""

Stein_GAN: batch_script.py

Created on 8/6/18 12:30 PM

@author: Hanxi Sun

"""

import os


EXP = "Compare_KSD"
SCRIPT_DIR = os.getcwd() + "/job_script/" + EXP + "/"
SERVER_DIR = "/home/sun652/Stein_GAN/"
if not os.path.exists(SCRIPT_DIR):
    os.makedirs(SCRIPT_DIR)

X_dim_lst = [3, 5, 7]
mu_dist_lst = [3, 5, 7]


def job_str(X_dim, mu_dist):
    return "_X_dim={0:02d}_mu_dist={1:2.2f}".format(X_dim, mu_dist)


############################################################
#                       Job Scripts                        #
############################################################

JOB_PREFIX = "JOB"
JOB_DIR = SERVER_DIR + "jobs/" + EXP + "/"
BATCH_SUBMIT = SCRIPT_DIR + "batch_submit.sh"
PYTHON_FILE = SERVER_DIR + "scr/" + EXP + ".py"
WALLTIME = "00:10:00"

PBS = ("#PBS -V \n" +
       "#PBS -q standby \n" +
       "#PBS -l walltime=" + WALLTIME + " \n\n")

submit = open(BATCH_SUBMIT, 'w')

submit.write("#!/bin/sh\n\n")

for X_dim in X_dim_lst:
    for mu_dist in mu_dist_lst:
        job_name = JOB_PREFIX + job_str(X_dim, mu_dist)
        job = open(SCRIPT_DIR + job_name + ".txt", 'w')
        job.write("#!/bin/sh -l\n" +
                  "# FILENAME: " + job_name + "\n\n" +
                  PBS +
                  "python " + PYTHON_FILE + " {0} {1} \n\n".format(X_dim, mu_dist))
        job.close()
        submit.write("qsub " + JOB_DIR + job_name + ".txt \n")
submit.close()


############################################################
#                       Delete Jobs                        #
############################################################
id0, id1 = 7518179, 7518191
for i in range(id0, id1+1):
    print("qdel {}".format(i))










