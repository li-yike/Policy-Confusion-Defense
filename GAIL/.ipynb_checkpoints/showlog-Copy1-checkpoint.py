import argparse
import pandas as pd
import h5py
import json
import numpy as np
import random
import matplotlib.patches as mpatches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfiles', type=str, nargs='+')
    parser.add_argument('--fields', type=str, default='trueret')
    parser.add_argument('--noplot', action='store_true')
    parser.add_argument('--plotfile', type=str, default=None)
    parser.add_argument('--range_end', type=int, default=None)
    args = parser.parse_args()
    assert len(set(args.logfiles)) == len(args.logfiles), 'Log files must be unique'

    fields = args.fields.split(',')

    # Load logs from all files
    fname2log = {}
    for fname in args.logfiles:
        with pd.HDFStore(fname, 'r') as f:
            assert fname not in fname2log
            df = f['log']
            df.set_index('iter', inplace=True)
            fname2log[fname] = df.loc[:args.range_end, fields]

    # 从中随机挑选对应个轨迹的颜色
    colors = {}
    color = ['firebrick']
    count = 0
    for fname,df in fname2log.items():
        colors[fname] = color[count]
        count += 1
    # Print stuff
    print(colors)
    if not args.noplot or args.plotfile is not None:
        import matplotlib
        if args.plotfile is not None:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt; plt.style.use('ggplot')

        ax = None
        handle = []
        for fname, df in fname2log.items():
            with pd.option_context('display.max_rows', 9999):
                print(fname)
                print(df[-1:])
                
            if 'vf_r2' in fields:
                df['vf_r2'] = np.maximum(0,df['vf_r2'])

            label = fname.split("-")[-1].split(".h5")[0]

            patch = mpatches.Patch(color=colors[fname],label=label)
            handle.append(patch)
            if ax is None:
                ax = df.plot(subplots=True, title=fname,color=colors[fname])
            else:
                df.plot(subplots=True, ax=ax, color=colors[fname],legend=False)

        plt.legend(handles=handle,bbox_to_anchor=(1.05, 1.0), loc='upper left')
        for i in range(len(fields)):
            ax[i].set_ylabel(fields[i])

        if args.plotfile is not None:
            plt.savefig(args.plotfile, bbox_inches='tight', dpi=200)
            print(args.plotfile)

        plt.show()
        plt.savefig('/root/code/imitation/figure.png',dpi=300) #save


if __name__ == '__main__':
    main()
