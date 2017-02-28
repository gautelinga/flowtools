import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Convert sequence of images to video.")
    parser.add_argument("folder", type=str, help="Folder with png files.")
    parser.add_argument("-framerate", type=int, help="Framerate.", default=25)
    args = parser.parse_args()

    os.system("ffmpeg -framerate {framerate} "
              "-start_number 126 "
              "-i {folder}/uz_%d.png "
              "-qscale:v 0 "
              "{folder}/uz.mp4".format(
                  folder=args.folder,
                  framerate=args.framerate))


if __name__ == "__main__":
    main()
