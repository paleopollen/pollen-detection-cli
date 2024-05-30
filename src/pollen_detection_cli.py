import argparse
import logging

from datetime import datetime
from pollen_detector import PollenDetector
import torch.multiprocessing as mp


class PollenDetectionCLI:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Process PNG image stacks and detect pollen grains.")
        self.args = None

        # Add arguments

        # Model directory path
        self.parser.add_argument("--model-path", "-m", type=str, dest="model_file_path", nargs='?', default=None,
                                 required=True, help="Full path of the trained model.")
        # Crop images directory path
        self.parser.add_argument("--crops-dir", "-c", type=str, dest="crops_dir_path", nargs='?', default=None,
                                 required=True, help="Full path of the directory containing the cropped image files.")
        # Detections output directory path
        self.parser.add_argument("--detections-dir-prefix", "-d", type=str, dest="detections_dir_path_prefix",
                                 nargs="?",
                                 default="detections", required=False,
                                 help="Full path prefix of the directory to store the detection results.")
        # Run in parallel
        self.parser.add_argument("--parallel", "-p", action="store_true", dest="parallel", default=False,
                                 help="Run the detection in parallel.")
        # Number of processes
        self.parser.add_argument("--num-processes", "-n", type=int, dest="num_processes", nargs="?", default=8,
                                 help="Number of processes to use for parallel processing.")
        # Number of data loading workers 0 or greater
        self.parser.add_argument("--num-workers", "-w", type=int, dest="num_workers", nargs="?", default=0,
                                 help="Number of data loading workers to use for parallel processing.")
        # Batch size
        self.parser.add_argument("--batch-size", "-b", type=int, dest="batch_size", nargs="?", default=8,
                                 help="Batch size for parallel processing.")
        # Shuffle dataset
        self.parser.add_argument("--shuffle", "-s", action="store_true", dest="shuffle", default=False,
                                 help="Shuffle the dataset.")
        # CPU only
        self.parser.add_argument("--cpu", action="store_true", dest="cpu", default=False,
                                 help="Run the detection on CPU only.")
        # Verbose
        self.parser.add_argument("--verbose", "-v", action="store_true", dest="verbose", default=False,
                                 help="Display more details.")

    def parse_args(self):
        self.args = self.parser.parse_args()

    def print_args(self):
        if self.args.verbose:
            print("\nArguments\tValues")
            print("======================")
            for key, value in self.args.__dict__.items():
                print(key, ":", value)
            print("======================\n")


if __name__ == '__main__':
    start_time = datetime.now()
    logging.basicConfig(format='%(asctime)s %(levelname)-7s : %(name)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger("pollen_detection_cli.py")

    logger.info("Starting Pollen Detection CLI")
    cli = PollenDetectionCLI()
    cli.parse_args()
    cli.print_args()

    pollen_detector = PollenDetector(cli.args.model_file_path, cli.args.crops_dir_path,
                                     cli.args.detections_dir_path_prefix, cli.args.num_processes, cli.args.num_workers,
                                     cli.args.batch_size, cli.args.cpu, cli.args.shuffle, cli.args.verbose)

    pollen_detector.generate_dbinfo()
    pollen_detector.initialize_dataset()
    pollen_detector.initialize_model()
    if cli.args.parallel:
        mp.set_start_method('spawn',
                            force=True)  # Ref: https://github.com/pytorch/pytorch/issues/804#issuecomment-1839388574
        pollen_detector.process_parallel()
        pollen_detector.process_pollen_detections()
    else:
        pollen_detector.initialize_data_loader()
        pollen_detector.find_potential_pollen_detections()
        pollen_detector.process_pollen_detections()

    logger.info("Stopping Pollen Detection CLI")
    logger.info("Total execution time: {}".format(datetime.now() - start_time))
