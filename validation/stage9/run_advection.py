"""Stage 9 benchmark: rigid rotation plus general affine trajectory."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from .trajectory import rigid_rotation_comparison, affine_flow


def _write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='validation/results/stage9')
    parser.add_argument('--plots',action='store_true')
    args=parser.parse_args(argv)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    results=[rigid_rotation_comparison(),affine_flow(pic_fraction=1.),affine_flow(pic_fraction=0.)]
    all_pass=all(r.passed for r in results)
    payload={'suite':'GeoPyMPM Stage 9 - affine-exponential advection',
             'passed':all_pass,
             'cases':[{'name':r.name,'passed':r.passed,'metrics':r.metrics,
                       'notes':r.notes} for r in results]}
    (output/'9_summary.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    for index,r in enumerate(results):
        name=('9_rotation_history.csv','9_affine_PIC_history.csv','9_affine_FLIP_history.csv')[index]
        _write_csv(output/name,r.history)
        print(f"{'PASS' if r.passed else 'FAIL'} | {r.name} ({'PIC' if index==1 else 'FLIP' if index==2 else 'reference comparison'})")
        for key,value in r.metrics.items(): print(f'    {key}: {value:.10g}')
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        r=results[0]
        fig,ax=plt.subplots(figsize=(7,4.3))
        ax.plot([h['time'] for h in r.history],
                [h['euler_position_error_m'] for h in r.history],label='Euler position')
        ax.plot([h['time'] for h in r.history],
                [h['affine_exponential_position_error_m'] for h in r.history],label='Affine exponential')
        ax.set(xlabel='Time (s)',ylabel='Max absolute position error (m)',
               title='Stage 9: prescribed rigid rotation')
        ax.grid(True,alpha=.3);ax.legend();fig.tight_layout()
        fig.savefig(output/'9_rotation_position_error.png',dpi=150)
        plt.close(fig)
    print(f'Outputs: {output.resolve()}')
    return 0 if all_pass else 1


if __name__=='__main__':
    raise SystemExit(main())
