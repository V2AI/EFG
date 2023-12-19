def list_oss_dir(oss_path, client, with_info=False):
    """
    Loading files from OSS
    """
    files_iter = client.get_file_iterator(oss_path)
    if with_info:
        file_list = {p: k for p, k in files_iter}
    else:
        file_list = [p for p, k in files_iter]
    return file_list
