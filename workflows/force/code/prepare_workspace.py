"""Create output directories without touching retained result files."""
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    args = parser.parse_args()
    for name in ('data/external', 'data/cache', 'results', 'qa'):
        (args.workspace / name).mkdir(parents=True, exist_ok=True)
    print('Workspace directories prepared. Existing files were preserved.')


if __name__ == '__main__':
    main()
