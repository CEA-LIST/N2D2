from os.path import dirname, abspath
from os.path import join as join_path
print(dirname(abspath(__file__)))
print(join_path(dirname(abspath(__file__)), "examples", "graph_example.py"))
# import n2d2


# a = n2d2.cells.Block([n2d2.cells.Fc(1,1, name='KO'),
#                     n2d2.cells.Fc(1,1, name='OK')])

# print(a[0])

# print(a["OK"])
