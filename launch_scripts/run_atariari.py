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
args = parser.parse_args()
base_cmd = args.base_cmd
if args.unkillable:
    file = "unkillable_mila_cluster.sl"
elif args.main:
    file = "main_mila_cluster.sl"
elif args.eval:
    file = "just_eval_mila_cluster.sl"
else:
    file = "mila_cluster.sl"
ss= "launch_scripts/" + file

# module = "scripts.run_probe"
run_args = [base_cmd, ss]
#run_args.extend(sys.argv[1:])

envs = ['asteroids',
'berzerk',
'bowling',
'boxing',
'breakout',
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
#'space_invaders',
'tennis',
'venture',
'video_pinball',
'yars_revenge']

#envs = ['pong', 'space_invaders', 'ms_pacman']
if args.envs != "None":
    envs = args.envs

if args.eval:
    assert args.ids != None, "specify the ids of the runs"
    for id in args.ids:
        sargs = run_args + ["--wandb-tr-id", id]
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
            if args.method == "stdim":
                sargs.extend(['--embedding-dim', '256'])
            else:
                sargs.extend(['--embedding-dim', '32', "--num-slots", "8"])

        elif args.regime == "cswm":
            sargs.extend(['--regime', 'cswm', "--color", '--num-episodes', '1000', '--embedding-dim', '4', '--action-dim', '6', '--num-slots', '3',
             '--copy-action',"--noop-max", "0", "--num-frame-stack", "2", "--screen-size", "50", "50", '--frameskip', '4',
                          '--hidden-dim', '512', "--max-episode-steps", "11"])
            if env in ["space_invaders", "pong"]:
                sargs.extend(["--num-slots", "3"])
                if env == "pong":
                    sargs.extend([ '--crop', '35', '190'])
                    #"--crop", "35", "190", "--warmstart", "58",
                else:
                    sargs.extend([ "--crop", '30', '200'])
                    #"--crop", "30", "200", "--warmstart", "50",
            else:
                sargs.extend(["--num-slots", "5"])
            if args.method == "cswm":
                sargs.extend([ '--epochs', '200',  '--lr', '5e-4', '--batch-size', '1024'])
            else:
                sargs.extend(['--lr', '5e-4'])
                sargs.extend(['--epochs', str(args.epochs), '--batch-size', str(args.batch_size)])



        sargs.extend(["--method", args.method])

        print(" ".join(sargs))
        subprocess.run(sargs)
