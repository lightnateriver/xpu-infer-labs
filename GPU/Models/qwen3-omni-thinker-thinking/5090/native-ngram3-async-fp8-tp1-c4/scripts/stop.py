"""Stop only the PID/session identified by this package's receipt."""
import argparse
import json
from pathlib import Path
from ops import ROOT, stop_owned

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, default=ROOT/'.local/service/receipt.json')
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    stop_owned(receipt)
    print('Owned service stopped; selected GPU process and API listener released.')
