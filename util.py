def mkdir(filepath_or_directory):
    '''
    make directory if not exist
    '''
    import os

    dirname = os.path.dirname(filepath_or_directory)
    if dirname:
        if not os.path.exists(dirname):
            os.makedirs(dirname)