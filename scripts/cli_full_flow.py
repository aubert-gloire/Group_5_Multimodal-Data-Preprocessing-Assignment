"""
Combined CLI that enforces the auth flow:
 - Simulated face recognition (allow/deny)
 - If face allowed, simulate voice approval (allow/deny)
 - If both pass, run product prediction using the trained pipeline

Usage examples:
  python scripts/cli_full_flow.py --csv-row ../Dataset/merged_customer_data.csv --row-index 0 --face-allow --voice-allow
  python scripts/cli_full_flow.py --interactive --face-deny  # will deny at face check
"""
import argparse
import sys
import pandas as pd
from pathlib import Path
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PREDICT_SCRIPT = os.path.join(ROOT, 'scripts', 'predict_product.py')


def simulate_face(auth_allow: bool) -> bool:
    print('Simulating face recognition ->', 'ALLOW' if auth_allow else 'DENY')
    return auth_allow


def simulate_voice(auth_allow: bool) -> bool:
    print('Simulating voice approval ->', 'ALLOW' if auth_allow else 'DENY')
    return auth_allow


def run_predict_interactive(require_face, require_voice, force_unauth):
    # Build argument list and call predict_product main via subprocess-like exec
    import runpy
    args = ['--interactive']
    if require_face:
        args.append('--require-face')
    if require_voice:
        args.append('--require-voice')
    if force_unauth:
        args.append('--force-unauthorized')
    sys.argv = ['predict_product.py'] + args
    runpy.run_path(os.path.join(ROOT, 'scripts', 'predict_product.py'), run_name='__main__')


def run_predict_csv(csv_path, row_index, require_face, require_voice, force_unauth):
    import runpy
    args = ['--csv-row', csv_path, '--row-index', str(row_index)]
    if require_face:
        args.append('--require-face')
    if require_voice:
        args.append('--require-voice')
    if force_unauth:
        args.append('--force-unauthorized')
    sys.argv = ['predict_product.py'] + args
    runpy.run_path(os.path.join(ROOT, 'scripts', 'predict_product.py'), run_name='__main__')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true', help='Run interactive prediction (prompts user)')
    parser.add_argument('--csv-row', type=str, help='CSV path to take a row from')
    parser.add_argument('--row-index', type=int, default=0)
    parser.add_argument('--face-allow', action='store_true')
    parser.add_argument('--face-deny', action='store_true')
    parser.add_argument('--voice-allow', action='store_true')
    parser.add_argument('--voice-deny', action='store_true')
    args = parser.parse_args()

    # Determine auth outcomes
    face_ok = True if args.face_allow else False if args.face_deny else True
    voice_ok = True if args.voice_allow else False if args.voice_deny else True

    # Simulate
    if not simulate_face(face_ok):
        print('Access denied at face recognition stage.')
        sys.exit(1)

    if not simulate_voice(voice_ok):
        print('Access denied at voice approval stage.')
        sys.exit(1)

    # If both pass, run prediction
    if args.interactive:
        run_predict_interactive(require_face=False, require_voice=False, force_unauth=False)
    elif args.csv_row:
        run_predict_csv(args.csv_row, args.row_index, require_face=False, require_voice=False, force_unauth=False)
    else:
        print('No prediction mode selected. Use --interactive or --csv-row')
