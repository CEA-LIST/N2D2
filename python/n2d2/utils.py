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

class ConfigSection(dict):
    def __init__(self, **kwargs):
        self._config_params = kwargs

    def get(self):
        return self._config_params

    def __str__(self):
        print(self._config_params)

def convert_to_INI(path, database, provider, deepnet, target):
    file = open(path+".ini", 'w')
    output = "DefaultModel=" + deepnet.get_model() + "\n\n"
    output += database.convert_to_INI_section() + "\n"
    output += provider.convert_to_INI_section() + "\n"
    output += deepnet.convert_to_INI_section() + "\n"
    output += target.convert_to_INI_section() + "\n"
    file.write(output)
    file.close()


def download_model(url, install_dir, target_dir):

    def progress(chunks_so_far, chunk_size, total_size):
        size_so_far = min(total_size, chunks_so_far * chunk_size)
        print("Downloaded %d of %d bytes (%3.1f%%)\r" % (size_so_far, total_size, 100.0 * float(size_so_far) / total_size), end="r")
        if size_so_far == total_size:
            sys.stdout.write("\n")
        sys.stdout.flush()

    (baseUrl, fileName) = url.rsplit('/', 1)
    target = os.path.join(install_dir, target_dir)
    print(target)
    if not os.path.exists(target):
        os.makedirs(target)
    target = os.path.join(target, fileName)
    if not os.path.exists(target):
        print(url + " -> " + target)
        urllib.request.urlretrieve(baseUrl + "/"
                           + urllib.parse.quote(fileName), target, progress)
        if fileName.endswith(".tar.gz") or fileName.endswith(".tar.bz2") \
                or fileName.endswith(".tar"):
            raw = tarfile.open(target)
            for m in raw.getmembers():
                raw.extract(m, os.path.dirname(target))
        elif fileName.endswith(".gz"):
            raw = gzip.open(target, 'rb').read()
            open(os.path.splitext(target)[0], 'wb').write(raw)
        elif fileName.endswith(".zip"):
            raw = zipfile.ZipFile(target, 'r')
            raw.extractall(os.path.dirname(target))




