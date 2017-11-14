import os, subprocess as sp, sys
import multiprocessing as mp
from queue import Queue
from threading import Thread
import threading
import argparse

path_to_stipdet = '/home/emanon/Desktop/Ac_BL/code/stip-2.0-linux/bin/stipdet'
path_to_stipdet_lib = '/home/emanon/Desktop/Ac_BL/code/stip-2.0-linux/lib'
# Contains links to export the LD_LIBRARY_PATH correctly

def log(*args):
    print(*args)


def calc_stip_features(path_to_videos, filename, path_to_output_file, ext=None, start_frame=None, end_frame=None,
                       overwrite=True):
    """
    Calculates the STIP features for a specified set of input files and stores them in the output location.
    :param path_to_videos: Directory containing the training dataset to be computed
    :param filename: Input filename of video
    :param path_to_output_file: Output file location containing the STIP features.
    :param ext: Extension of video, if None is provided, then "avi" is assumed
    :param start_frame: (Optional) Start frame of the video
    :param end_frame: (Optional) End frame of the video
    :return:
    """

    if not overwrite and os.path.isfile(path_to_output_file):
        # Skip if file already present
        log("Ignoring file {}{}, because overwrite is disabled".format(filename, ext))
        return

    if start_frame and end_frame:
        video_list_file = filename + ".temp." + start_frame + "-" + end_frame + ".txt"
    else:
        video_list_file = filename + ".temp.txt"

    with open(os.path.join(path_to_videos, video_list_file), "w") as video_list:
        if start_frame is not None and end_frame is not None:
            video_list.write(filename + " " + start_frame + " " + end_frame + "\n")
        else:
            video_list.write(filename + "\n")

    ext_str = " -ext " + ext if ext else ""

    args = path_to_stipdet + " -i " + \
           os.path.join(path_to_videos, video_list_file) + ext_str + " -vpath " + path_to_videos + " -o " + \
           path_to_output_file + " -det harris3d -vis no -stdout no"

    process = ["/bin/bash", "-c", args]

    with sp.Popen(process, env=dict(os.environ, LD_LIBRARY_PATH=path_to_stipdet_lib), stdout=sp.DEVNULL) as p:
        try:
            log("Running Stipdet on {}{}{}".format(filename, ext,
                                                   " with frames " + start_frame + "-" + end_frame if start_frame else ""))
            retcode = p.wait()
            if retcode:
                cmd = "stipdet"
                raise sp.CalledProcessError(retcode, cmd)
        except:
            p.kill()
            p.wait()
        finally:
            os.remove(os.path.join(path_to_videos, video_list_file))


def ffmpeg_sample(input_file, output_file, sample_rate, overwrite=True):
    """

    :param input_file:
    :param output_file:
    :param sample_rate: Number of frames to sample, i.e. every 5, 10 frames, etc.
    :param overwrite: Overwrite, if file already exists
    :return:
    """
    overwrite_str = " -y " if overwrite else " -n "
    cmd = "ffmpeg -v warning -stats" + overwrite_str + \
          " -threads 12 -i " + input_file + " -vf \"select=\'not(mod(n\," + str(sample_rate) +  "))\'\" " \
          "-b:v 2M -crf 18 -c:v \'vp9\' '" + output_file + "'"

    process = ["/bin/bash", "-c", cmd]

    with sp.Popen(process, stderr=sp.DEVNULL, stdout=sp.DEVNULL) as p:
        retcode = p.wait()
        if retcode:
            p.kill()
            p.wait()
            cmd = "ffmpeg"
            raise sp.CalledProcessError(retcode, cmd)
        else:
            log("Finished Sampling {}".format(input_file))




def worker(queue, run):
    """Process files from the queue."""
    for args in iter(queue.get, None):
        try:
            run(*args)
        except Exception as e: # catch exceptions to avoid exiting the thread prematurely
            print('{} failed: {}'.format(args, e), file=sys.stderr)


def start_stip_in_parallel(queue, number_of_process=None):
    start_processes_in_parallel(queue, calc_stip_features, number_of_process)

def start_processes_in_parallel(queue, func, number_of_process=None):
    """
        Starts threads to run the function with processes in parallel
        :param queue: Queue of tasks i.e. files to process, also the arguments to run()
        :param number_of_process: If none is provided, the total number of CPU Cores - 1 is taken.
        :return: Nothing is returned.
    """
    if not number_of_process:
        number_of_process = mp.cpu_count() - 1

        # start threads
    threads = [Thread(target=worker, args=(queue, func)) for _ in range(number_of_process)]
    for t in threads:
        t.daemon = True  # threads die if the program dies
        t.start()
    for _ in threads: queue.put_nowait(None)  # signal no more files
    for t in threads: t.join()  # wait for completion


def main(path_to_videos, overwrite=True, num_proc=None):

    # path_to_videos = '/home/emanon/Desktop/Ac_BL/data/Judo/Top/'  # '/home/emanon/Desktop/Ac_BL/data/Judo/Test/'
    path_to_features = os.path.join(path_to_videos, "descr/")

    # populate files
    if not os.path.exists(path_to_features): os.mkdir(path_to_features)

    q = Queue()
    for file in os.listdir(path_to_videos):
        if os.path.isfile(os.path.join(path_to_videos, file)):
            filename, ext = os.path.splitext(file)
            q.put_nowait((path_to_videos, filename, os.path.join(path_to_features, filename + ".txt"), ext,
                          None, None, overwrite))

    start_stip_in_parallel(q, num_proc)


def sample(path_to_videos, sample_rate='5', overwrite=True, num_proc=None):
    # path_to_videos = '/home/emanon/Desktop/Ac_BL/data/Judo/Side/'  # '/home/emanon/Desktop/Ac_BL/data/Judo/Test/'
    path_to_output = path_to_videos + "sampled/"

    # populate files
    if not os.path.exists(path_to_output): os.mkdir(path_to_output)

    q = Queue()
    for file in os.listdir(path_to_videos):
        path_to_file = os.path.join(path_to_videos, file)
        if os.path.isfile(path_to_file):
            q.put_nowait((path_to_file, os.path.join(path_to_output, file), sample_rate, overwrite))

    start_processes_in_parallel(q, ffmpeg_sample, num_proc)

def ensure_trailing_slash(str):
    return str + "/" if str[-1] is not '/' else str

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--enable-sample', dest='enable_sample', action='store_const', const=True, default=False, help='Sample the Directory')
    parser.add_argument('-s', '--sample-rate', dest='sample_rate', nargs=1, default='5', help='Sample Rate during sampling')
    parser.add_argument('-y', '--overwrite', dest='overwrite', action='store_const', default=False, const=True, help='Enable Overwriting of files.')
    parser.add_argument('-n', '--proc', dest='proc', default=[mp.cpu_count() - 1], nargs=1, help='Number of processes to run in parallel', type=int, metavar='N')
    parser.add_argument('PATH', type=str, help='Directory with respective files to be processed')
    args = parser.parse_args()
    if args.enable_sample:
        sample(ensure_trailing_slash(args.PATH), args.sample_rate[0], args.overwrite, args.proc[0])
    else:
        main(ensure_trailing_slash(args.PATH), args.overwrite, args.proc[0])
