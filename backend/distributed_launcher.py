"""
Distributed Launcher
Small CLI helper to inspect the distributed-engineering stack.
"""

from distributed_engine import DistributedEngine


def main():
    engine = DistributedEngine()
    summary = engine.get_summary()
    print('Distributed engineering summary:')
    for key, value in summary.items():
        print(f'- {key}: {value}')


if __name__ == '__main__':
    main()
