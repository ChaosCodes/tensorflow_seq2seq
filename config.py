import argparse
import sys

def load_arguments():
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument('--gpu_id',
							type=int,
							default=0)
