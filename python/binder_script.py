import os


def activation_doc():
    n=[]
    f=[]
    files = os.listdir("../src/python/Activation")
    for file in files:
        act_name = file.split('.')[0].split('_')[1:]
        w = act_name[0]
        for i in act_name[1:]:
            w += '_' + i
        if len(act_name) != 1:
            f.append(w)
        else:
            n.append(w)

    print("Activation\n----------\n")

    for i in n:
        print(i + '\n' + ('~' * len(i)) + '\n\n' + ".. autoclass:: N2D2."+i+'\n\t:members:\n')

    print("Activation_Frame\n----------------\n")

    for i in range(0,len(n)-1, 2):
        print(f[i] + '\n' + ('~' * len(f[i])) + "\n\n.. autoclass:: N2D2."+f[i]+'_float\n\t:members:\n'+
        ".. autoclass:: N2D2."+f[i]+'_double\n\t:members:\n'+
        '.. autoclass:: N2D2.'+f[i+1]+'_float\n\t:members:\n' +
        '.. autoclass:: N2D2.'+f[i+1]+'_double\n\t:members:\n')

def check_transform_templated(path):
    file = open(path, 'r')
    for line in file:
        if 'template' in line:
            return True
    return False

    file.close()



def database_doc():
    n=[]
    f=[]
    files = os.listdir("../src/python/Database")
    for file in files:
        db_name = file.split('.')[0].split('_')[1:]
        s = db_name[0]
        for i in db_name[1:]:
            s += "_" + i
        print(s + "\n" +("~" * len(s)) + "\n\n.. autoclass:: N2D2."+s+'\n\t:members:\n\t:inherited-members:\n')

        


def transformation_doc():
    frame = []
    cell = []
    spike = []
    files = os.listdir("../src/python/Cell")
    files.sort()
    for f in files:
        file = f.split(".")[0].split('_')[1:]
        s = file[0]
        for i in file[1:]:
            s+= '_'+i
        if 'Frame' in file:
            if check_transform_templated(("../src/python/Cell/" + f)):
                frame.append(s + '_float')
                frame.append(s + '_double')
            else:
                frame.append(s)
        elif 'Spike' in file:
            spike.append(s)
        else:
            cell.append(s)
    fichier = open("./pybind_output_to_remove", 'w')
    fichier.write("Cell\n----\n\n")
    for i in cell:
        fichier.write(i+"\n"+('~'*len(i))  +"\n\n.. autoclass:: N2D2."+i+"\n\t:members:\n\t:inherited-members:\n\n")

    fichier.write("\nFrame\n-----\n\n")
    for i in frame:
        fichier.write(i+"\n"+('~'*len(i))  +"\n\n.. autoclass:: N2D2."+i+"\n\t:members:\n\t:inherited-members:\n\n")

    fichier.write("\nSpike\n-----\n\n")
    for i in spike:
        fichier.write(i+"\n"+('~'*len(i))  +"\n\n.. autoclass:: N2D2."+i+"\n\t:members:\n\t:inherited-members:\n\n")
    fichier.close()

def cell_bind():
    n=[]
    f=[]
    cpp_files = os.listdir("../src/Cell")
    py_files = os.listdir("../src/python/Cell")
    declare = []
    call = []
    for file in cpp_files:
        if file.split('.')[-1] != 'cu' and ('pybind_') + file not in py_files:
            print("Creating ", '../src/python/Cell/pybind_'+ file)
            os.system("touch ../src/python/Cell/pybind_" + file)
            cell_name = file.split('.')[0]
            print(cell_name)
            declare.append("void init_"+cell_name+"(py::module&);")
            call.append("init_"+cell_name+"(m);")
    
    fichier = open("./pybind_output_to_remove", 'w')
    for i in declare:
        fichier.write(i + '\n')
    fichier.write('\n')
    for i in call:
        fichier.write(i + '\n')
    fichier.close()

database_doc()
