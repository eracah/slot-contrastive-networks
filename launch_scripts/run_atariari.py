import subprocess
import sys
import argparse
import os
print(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--unkillable", action='store_true', default=False)
parser.add_argument("--main", action='store_true', default=False)
parser.add_argument("--regime", type=str, default="stdim", choices=["stdim", "cswm"],
                    help="whether to use the encoder and dataloader from stdim or from cswm")
parser.add_argument("--base-cmd", type=str, default="sbatch", choices=["sbatch", "bash"])
parser.add_argument('--method', type=str, default='scn', help='Method to use for training representations (default: scn')
parser.add_argument('--envs', type=str, nargs="+", default='None')
parser.add_argument('--epochs',type=int, default= 600)
parser.add_argument('--batch-size', default=1024, type=int)
parser.add_argument("--eval", action='store_true', default=False)
parser.add_argument("--ids", type=str, nargs="+", default="None")
args, other_args  = parser.parse_known_args()
base_cmd = args.base_cmd
if args.eval:
    if args.unkillable:
        file =  "unkillable_just_eval_mila_cluster.sl"
    else:
        file = "just_eval_mila_cluster.sl"
else:
    if args.unkillable:
        file = "unkillable_mila_cluster.sl"
    elif args.main:
        file = "main_mila_cluster.sl"
    else:
        file = "mila_cluster.sl"
ss= "launch_scripts/" + file

# module = "scripts.run_probe"
run_args = [base_cmd, ss]
#run_args.extend(sys.argv[1:])

envs = [#'asteroids',
'berzerk',
'bowling',
'boxing',
#'breakout',
'demon_attack',
#'freeway',
'frostbite',
'hero',
'montezuma_revenge',
#'ms_pacman',
'pitfall',
#'pong',
'private_eye',
'qbert',
'riverraid',
'seaquest',
'space_invaders',
'tennis',
'venture',
'video_pinball',
'yars_revenge']

#envs = ['freeway', 'ms_pacman','breakout', 'asteroids']
if args.envs != "None":
    envs = args.envs

if args.eval:
    assert args.ids != None, "specify the ids of the runs"
    for id in args.ids:
        sargs = run_args + ["--id", id]
        print(" ".join(sargs))
        subprocess.run(sargs)

else:
    suffix = "NoFrameskip-v4"
    for i,env in enumerate(envs):

        names = env.split("_")
        name = "".join([s.capitalize() for s in names])
        sargs = run_args + ["--env-name"]

        sargs.append(name + suffix)

        sargs.extend(["--wandb-proj", "coors-production"])
        if args.regime == "stdim":
            sargs.extend(["--regime", "stdim", "--color",  "--lr",  "3e-4",  "--num-frames", "100000", "--batch-size", "128", "--epochs", "100"])



        sargs.extend(["--method", args.method])
        sargs.extend(other_args)

        print(" ".join(sargs))
        subprocess.run(sargs)
