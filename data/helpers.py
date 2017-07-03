"""
Helping functions called by the pipeline modules.
"""

########################################################
def get_lines_in_file(file_path, read=False):
    """
    Reads file lines and returns number of lines, and optionally the lines list

    :param file_path
    :param read: boolean to alloe=w return of lines contents not just the number of lines
    :return: number of lines, and optionally the lines list
    """
    nLines = 0
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            nLines += 1
            if read:
                lines.append(line.strip())
    return lines, nLines

########################################################
def is_png(filename):
    """Determine if a file contains a PNG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a PNG.
    """
    return '.png' in filename


########################################################
def is_jpg(filename):
    """Determine if a file contains a PNG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a JPG.
    """
    return '.jpg' in filename
