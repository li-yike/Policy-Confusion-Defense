import os, os.path
import argparse
import transformer

parser = argparse.ArgumentParser()
parser.add_argument('envs', type=str, nargs='+')
parser.add_argument('--outdir', type=str,required=True)
args = parser.parse_args()

outdir = args.outdir
cmd_template = 'python scripts/run_rl_mj.py --env_name {env} --tiny_policy --min_total_sa 5000 --sim_batch_size 1 --max_iter 501 --log {out}'

print(outdir)
try: 
    os.mkdir(outdir)
except OSError: 
    pass

for env in args.envs:
    cmd = cmd_template.format(env=env, out=os.path.join(outdir, 'log_'+env+'.h5'))
    print(cmd)
    os.system(cmd)
    print('\n\n\n')
