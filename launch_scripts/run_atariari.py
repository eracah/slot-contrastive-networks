import subprocess
import sys
base_cmd = "sbatch"
ss= "launch_scripts/mila_cluster.sl"
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
    sargs.extend(["--eval_args", "--wandb-proj", "coors-production"])

    print(" ".join(sargs))
    subprocess.run(sargs)
