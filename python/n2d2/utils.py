"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)

    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software.  You can  use,
    modify and/ or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".

    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
"""

import sys
import os
import urllib.request
import urllib.parse
import tarfile
import gzip, zipfile
from collections import UserDict


# At the moment ConfigSection is simply a dictionary
class ConfigSection(UserDict):
    pass


def download_model(url, install_dir, dir_name):

    def progress(chunks_so_far, chunk_size, total_size):
        size_so_far = min(total_size, chunks_so_far * chunk_size)
        print("Downloaded %d of %d bytes (%3.1f%%)\r" % (size_so_far, total_size, 100.0 * float(size_so_far) / total_size), end="r")
        if size_so_far == total_size:
            sys.stdout.write("\n")
        sys.stdout.flush()

    (base_url, file_name) = url.rsplit('/', 1)
    target = os.path.join(install_dir, dir_name)
    print(target)
    if not os.path.exists(target):
        os.makedirs(target)
    target = os.path.join(target, file_name)
    print(target)
    if not os.path.exists(target):
        print(url + " -> " + target)
        urllib.request.urlretrieve(base_url + "/"
                           + urllib.parse.quote(file_name), target, progress)
        if file_name.endswith(".tar.gz") or file_name.endswith(".tar.bz2") \
                or file_name.endswith(".tar"):
            raw = tarfile.open(target)
            for m in raw.getmembers():
                raw.extract(m, os.path.dirname(target))
        elif file_name.endswith(".gz"):
            raw = gzip.open(target, 'rb').read()
            open(os.path.splitext(target)[0], 'wb').write(raw)
        elif file_name.endswith(".zip"):
            raw = zipfile.ZipFile(target, 'r')
            raw.extractall(os.path.dirname(target))




