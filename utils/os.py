""" Operation system utilities. """
import glob
import os
import shutil


def fresh_dir(dir_path):
    if os.path.exists(dir_path):
        for file_or_dir in glob.glob(dir_path):
            if os.path.isfile(file_or_dir):
                os.remove(file_or_dir)
            elif os.path.isdir(file_or_dir):
                shutil.rmtree(file_or_dir)
            else:
                pass
        return
    os.mkdir(dir_path)


def files_in_dir(dir_path, filter_fn=None, recurrent=False):
    if not os.path.isdir(dir_path):
        raise NotADirectoryError('%s is not a valid path to a directory.' % dir_path)

    all_files = []
    for path in map(lambda s: os.path.join(dir_path, s), os.listdir(dir_path)):
        if os.path.isdir(path):
            if recurrent:
                all_files.extend(files_in_dir(path, filter_fn=filter_fn, recurrent=True))
        else:
            all_files.append(path)
    return list(filter(filter_fn, all_files))


def require_dir(dir_path):
    if not os.path.exists(dir_path):
        raise NotADirectoryError('Directory not found: %s.' % dir_path)


def make_dir(dir_path, cover_if_exists=None):
    """ Make a directory.
        Arguments:
            dir_path: A `str`, path to directory.
            cover_if_exists: If true, cover the directory if already exists, or else,
                raise an error. Defaults to make new directory as appropriate.
        Raises:
            ValueError: If `cover_if_exists` is false and the directory already exists.
    """
    if os.path.exists(dir_path):
        if cover_if_exists is True:
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
        elif cover_if_exists is False:
            raise ValueError('Directory already exists: %s.' % dir_path)
    else:
        os.mkdir(dir_path)
