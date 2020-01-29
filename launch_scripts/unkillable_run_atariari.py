import subprocess
import sys
base_cmd = "sbatch"
ss= "launch_scripts/unkillable_mila_cluster.sl"
module = "scripts.run_probe"
args = [base_cmd, ss, module]
args.extend(sys.argv[1:])


#envs = ["pong", "freeway","ms_pacman"]
#envs = [ 'video_pinball',
 #'yars_revenge', 'space_invaders', 'private_eye', 'montezuma_revenge',
 #'ms_pacman', 'demon_attack']

#envs = ['asteroids',
#'berzerk',
#'bowling',
##'boxing',
#'breakout',
#'demon_attack',
#'freeway',
#'frostbite',
#'hero',
#'montezumai_revenge',
#'ms_pacman',
#'pitfall',
#'pong',
#'private_eye',
#'qbert',
#'riverraid',
#'seaquest',
#'space_invaders',
#'tennis',
#'venture',
#'video_pinball',
#'yars_revenge']
# envs =  ["asteroids", "freeway", "montezuma_revenge", 'berzerk', 'boxing',
#         'demon_attack', 'enduro', 'freeway', 'frostbite', 'hero',
#         'ms_pacman', 'pong', 'private_eye', 'qbert', 'riverraid',
#         'seaquest', 'solaris', 'space_invaders', 'venture', 'video_pinball',
#         'yars_revenge','breakout','pitfall','montezuma_revenge'
#         ]


#suffix = "NoFrameskip-v4"
#names = env.split("_")
#name = "".join([s.capitalize() for s in names])
 #   sargs = args + ["--env-name"]

#  sargs.append(name + suffix)

args.extend(["--wandb-proj", "coors-production"])
print(" ".join(args))
subprocess.run(args)
