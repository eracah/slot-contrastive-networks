import subprocess
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--unkillable", action='store_true', default=False)
args = parser.parse_args()
base_cmd = "sbatch"
if args.unkillable:
    file = "unkillable_mila_cluster.sl"
else:
    file = "mila_cluster.sl"
ss= "launch_scripts/" + file

# module = "scripts.run_probe"
args = [base_cmd, ss]
args.extend(sys.argv[1:])

envs = ['asteroids',
'berzerk',
'bowling',
'boxing',
'breakout',
'demon_attack',
'freeway',
'frostbite',
'hero',
'montezumai_revenge',
'ms_pacman',
'pitfall',
'pong',
'private_eye',
'qbert',
'riverraid',
'seaquest',
'space_invaders',
'tennis',
'venture',
'video_pinball',
'yars_revenge']

suffix = "NoFrameskip-v4"
for i,env in enumerate(envs):

    names = env.split("_")
    name = "".join([s.capitalize() for s in names])
    sargs = args + ["--env-name"]

    sargs.append(name + suffix)

    sargs.extend(["--wandb-proj", "coors-production"])

    print(" ".join(sargs))
    subprocess.run(sargs)
