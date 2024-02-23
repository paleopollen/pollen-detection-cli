import argparse
import logging
import torch

from pollen_detector import PollenDetector
from torch import multiprocessing as mp


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

        # # Run in parallel
        # self.parser.add_argument("--parallel", "-p", action="store_true", dest="parallel", default=False,
        #                          help="Run the detection in parallel.")

        # Number of processes
        self.parser.add_argument("--num-processes", "-n", type=int, dest="num_processes", nargs="?", default=8,
                                 help="Number of processes to use for parallel processing.")

        # Batch size
        self.parser.add_argument("--batch-size", "-b", type=int, dest="batch_size", nargs="?", default=8,
                                 help="Batch size for parallel processing.")

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
    logging.basicConfig(format='%(asctime)s %(levelname)-7s : %(name)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger("pollen_detection_cli.py")

    logger.info("Starting Pollen Detection CLI")
    cli = PollenDetectionCLI()
    cli.parse_args()
    cli.print_args()

    pollen_detector = PollenDetector(cli.args.model_file_path, cli.args.crops_dir_path,
                                     cli.args.detections_dir_path_prefix, cli.args.num_processes, cli.args.batch_size)

    # if cli.args.parallel:
    #     logger.info("Running in parallel mode")
    #     manager = mp.Manager()
    #     return_dict = manager.dict()
    #
    #     worker_loaders = torch.utils.data.random_split(dataset, [len(dataset) // num_workers] * num_workers)
    #
    #     processes = []
    #     for worker_id in range(cli.args.num_workers):
    #         p = mp.Process(target=inference_worker, args=(
    #             model, DataLoader(worker_loaders[worker_id], batch_size=10, worker_init_fn=worker_init_fn(worker_id)),
    #             worker_id, return_dict))
    #         processes.append(p)
    #         p.start()
    #
    #     for p in processes:
    #         p.join()
    # else:
    pollen_detector.generate_dbinfo()
    pollen_detector.initialize_data()
    pollen_detector.initialize_model()
    pollen_detector.process_crop_images()

    logger.info("Stopping Pollen Detection CLI")
