"""Replay the immutable longer-trained historical checkpoints with the same evaluator."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from workflows.sbi import p12a_historical_knot_comparison as replay

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['unet','graph'],required=True)
    p.add_argument('--rotation',type=int,default=0);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    replay.OLD=replay.ROOT/'p8_recovery_v1/convergence_extension_v1'
    replay.replay(a)
