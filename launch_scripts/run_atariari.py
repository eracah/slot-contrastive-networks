import subprocess
import sys
base_cmd = "sbatch"
ss= "launch_scripts/mila_cluster.sl"
module = "scripts.run_probe"
args = [base_cmd, ss, module]
args.extend(sys.argv[1:])


envs = ["pong", "freeway" ,"ms_pacman"]

# envs =  ["asteroids", "freeway", "montezuma_revenge", 'berzerk', 'boxing',
#         'demon_attack', 'enduro', 'freeway', 'frostbite', 'hero',
#         'ms_pacman', 'pong', 'private_eye', 'qbert', 'riverraid',
#         'seaquest', 'solaris', 'space_invaders', 'venture', 'video_pinball',
#         'yars_revenge','breakout','pitfall','montezuma_revenge'
#         ]


suffix = "NoFrameskip-v4"
for i,env in enumerate(envs):

    names = env.split("_")
    name = "".join([s.capitalize() for s in names])
    sargs = args + ["--env-name"]

    sargs.append(name + suffix)

    sargs.extend(["--wandb-proj", "coors-production"])

    print(" ".join(sargs))
    subprocess.run(sargs)
