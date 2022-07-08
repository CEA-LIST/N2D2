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
import gzip
import zipfile
from collections import UserDict
from inspect import getmro, signature
import functools
from n2d2.error_handler import WrongInputType

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


_objects_counter = {}

def generate_name(obj):
    """
    Function used to generate name of an object
    """
    name = obj.__class__.__name__
    if name in _objects_counter:
        _objects_counter[name] += 1
    else:
        _objects_counter[name] = 0
    name += "_"+str(_objects_counter[name])
    return name


def model_logger(model, path, log_dict=None):
    print("Logging the model at " + path)
    model.export_free_parameters(path)
    if log_dict:
        file = open(path + "log.txt", "w")
        for key, value in log_dict.items():
            file.write(key + str(value))
        file.close()

def _get_param_docstring(docstring):
    header = True
    param_docstring = ""
    for line in docstring.split("\n"):
        if header and ":param" in line:
            header=False
        if not header:
            param_docstring+=line + "\n"
    return param_docstring

def inherit_init_docstring():
    """Decorator to inherit the docstring of __init__.
    """
    def dec(obj):
        parents_docstring = "\n"
        for parent in getmro(obj):
            if "__init__" in dir(parent) \
                    and parent.__init__.__doc__ \
                    and parent is not obj:
                parents_docstring += _get_param_docstring(parent.__init__.__doc__)
        docstring = obj.__init__.__doc__ if obj.__init__.__doc__ else ""
        obj.__init__.__doc__ = docstring + parents_docstring
        return obj
    return dec

def add_docstring(doc_string):
    """Decorator to inherit the docstring of another function.
    The docstring header is conserved.
    A dictionnary of the parameter is made by parsing, the docstring of the function and the docstring to add.
    The docstring available in the function override the docstring to add.
    """
    def dec(func):
        header = ""
        flag_header=True
        param_dic = {}
        for line in doc_string.split("\n"):
            if flag_header:
                if ":param" in line:
                    flag_header = False
                else:
                    continue
            if ":param" in line:
                param_name = line.split(":")[1].replace("param ", "")
                param_desc = line.split(":")[2].lstrip(" ")
                if param_name not in param_dic:
                    param_dic[param_name] = [param_desc, ""]
                else:
                    param_dic[param_name][0] = param_desc
            if ":type" in line:
                param_name = line.split(":")[1].replace("type ", "")
                param_desc = line.split(":")[2].lstrip(" ")
                if param_name not in param_dic:
                    param_dic[param_name] = ["", param_desc]
                else:
                    param_dic[param_name][1] = param_desc
        flag_header = True
        for line in func.__doc__.split("\n"):
            if flag_header:
                if ":param" in line:
                    flag_header = False
                else:
                    header += line + "\n"
            if ":param" in line:
                param_name = line.split(":")[1].replace("param ", "")
                param_desc = line.split(":")[2].lstrip(" ")
                if param_name not in param_dic:
                    param_dic[param_name] = [param_desc, ""]
                else:
                    param_dic[param_name][0] = param_desc
            if ":type" in line:
                param_name = line.split(":")[1].replace("type ", "")
                param_desc = line.split(":")[2].lstrip(" ")
                if param_name not in param_dic:
                    param_dic[param_name] = ["", param_desc]
                else:
                    param_dic[param_name][1] = param_desc

        param_doc = ""
        for param_name, param_desc in param_dic.items():
            param_doc += f":param {param_name}: {param_desc[0]}\n:type {param_name}: {param_desc[1]}\n"
        func.__doc__ = header + param_doc
        return func
    return dec

def methdispatch(meth):
    """Mimic the behavior of `functools.singledispatchmethod` which is only available in python >= 3.8.
    https://docs.python.org/3/library/functools.html#functools.singledispatchmethod
    """
    dispatcher = functools.singledispatch(meth)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    functools.update_wrapper(wrapper, dispatcher)
    return wrapper



def check_types(f):
    """
    Decorator used to automatically check type of functions/methods.
    To do so we use type annoation avaialble since Python 3.5 https://docs.python.org/3/library/typing.html.
    """
    sig = signature(f)

    # Dictionary key : param name, value : annotation
    args_types = {p.name: p.annotation \
            for p in sig.parameters.values()}

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        bind = sig.bind(*args, **kwargs)
        obj_name = ""

        # Check if we are in a method !
        if "self" in sig.parameters:
            obj_name = f"{bind.args[0].__class__.__name__}."

        for value, typ in zip(bind.args, args_types.items()):
            if typ[1] != sig.empty and not isinstance(value, typ[1]):
                raise TypeError(f'In {obj_name}{f.__name__} : \"{typ[0]}\" parameter must be of type <{typ[1].__name__}> but is of type <{type(value).__name__}> instead.')
        return f(*args, **kwargs)
    return decorated