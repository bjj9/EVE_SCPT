import os
import shutil

from shutil import copytree, ignore_patterns

if __name__ == '__main__':
    source = 'E:/datasets/eve_dataset'
    destination = 'E:/datasets/eve_dataset_tiny'
    copytree(source, destination, ignore=ignore_patterns('basler.mp4', '_face.mp4',
                        'screen.mp4', 'webcam_c.mp4', 'webcam_l.mp4', 'webcam_r.mp4',
                        'webcam_c_eyes.mp4', 'webcam_l_eyes.mp4', 'webcam_r_eyes.mp4',))



# def copytree(src, dst, symlinks=False, ignore=shutil.ignore_patterns('.*', '_*')):
#     """
#     Copy Entire Folder
#     :param src: source path
#     :param dst: destination path
#     :param symlinks: optional
#     :param ignore: pass shutil.ignore_patterns('.*', '_*')
#     :return:
#     """
#     for item in os.listdir(src):
#         s = os.path.join(src, item)
#         d = os.path.join(dst, item)
#         if os.path.isdir(s):
#             shutil.copytree(s, d, symlinks, ignore)
#         else:
#             shutil.copy2(s, d)