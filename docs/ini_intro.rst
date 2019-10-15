Introduction
============

The INI file interface is the primary way of using N2D2. It is a simple,
lightweight and user-friendly format for specifying a complete DNN-based
application, including dataset instanciation, data pre-processing,
neural network layers instanciation and post-processing, with all its
hyperparameters.

Syntax
------

INI files are simple text files with a basic structure composed of
sections, properties and values.

Properties
~~~~~~~~~~

The basic element contained in an INI file is the property. Every
property has a name and a value, delimited by an equals sign (=). The
name appears to the left of the equals sign.

::

    name=value

Sections
~~~~~~~~

Properties may be grouped into arbitrarily named sections. The section
name appears on a line by itself, in square brackets ([ and ]). All
properties after the section declaration are associated with that
section. There is no explicit “end of section” delimiter; sections end
at the next section declaration, or the end of the file. Sections may
not be nested.

::

    [section]
    a=a
    b=b

Case sensitivity
~~~~~~~~~~~~~~~~

Section and property names are case sensitive.

Comments
~~~~~~~~

Semicolons (``;``) or number sign (``#``) at the beginning or in the
middle of the line indicate a comment. Comments are ignored.

::

    ; comment text
    a=a # comment text
    a="a ; not a comment" ; comment text

Quoted values
~~~~~~~~~~~~~

Values can be quoted, using double quotes. This allows for explicit
declaration of whitespace, and/or for quoting of special characters
(equals, semicolon, etc.).

Whitespace
~~~~~~~~~~

Leading and trailing whitespace on a line are ignored.

Escape characters
~~~~~~~~~~~~~~~~~

A backslash (``\``) followed immediately by EOL (end-of-line) causes the
line break to be ignored.

Template inclusion syntax
-------------------------

Is is possible to recursively include templated INI files. For example,
the main INI file can include a templated file like the following:

::

    [inception@inception_model.ini.tpl]
    INPUT=layer_x
    SIZE=32
    ARRAY=2 ; Must be the number of elements in the array
    ARRAY[0].P1=Conv
    ARRAY[0].P2=32
    ARRAY[1].P1=Pool
    ARRAY[1].P2=64

If the ``inception_model.ini.tpl`` template file content is:

::

    [{{SECTION_NAME}}_layer1]
    Input={{INPUT}}
    Type=Conv
    NbOutputs={{SIZE}}

    [{{SECTION_NAME}}_layer2]
    Input={{SECTION_NAME}}_layer1
    Type=Fc
    NbOutputs={{SIZE}}

    {% block ARRAY %}
    [{{SECTION_NAME}}_array{{#}}]
    Prop1=Config{{.P1}}
    Prop2={{.P2}}
    {% endblock %}

The resulting equivalent content for the main INI file will be:

::

    [inception_layer1]
    Input=layer_x
    Type=Conv
    NbOutputs=32

    [inception_layer2]
    Input=inception_layer1
    Type=Fc
    NbOutputs=32

    [inception_array0]
    Prop1=ConfigConv
    Prop2=32

    [inception_array1]
    Prop1=ConfigPool
    Prop2=64

The ``SECTION_NAME`` template parameter is automatically generated from
the name of the including section (before ``@``).

Variable substitution
~~~~~~~~~~~~~~~~~~~~~

``{{VAR}}`` is replaced by the value of the ``VAR`` template parameter.

Control statements
~~~~~~~~~~~~~~~~~~

Control statements are between ``{\%`` and ``\%}`` delimiters.

block
^^^^^

``{\% block ARRAY \%}`` ... ``{\% endblock \%}``

The ``#`` template parameter is automatically generated from the
``{\% block ... \%}`` template control statement and corresponds to the
current item position, starting from 0.

for
^^^

``{\% for VAR in range([START, ]END]) \%}`` ... ``{\% endfor \%}``

If ``START`` is not specified, the loop begins at 0 (first value of
``VAR``). The last value of ``VAR`` is ``END``-1.

if
^^

``{\% if VAR OP [VALUE] \%}`` ... ``[{\% else \%}]`` ...
``{\% endif \%}``

``OP`` may be ``==``, ``!=``, ``exists`` or ``not_exists``.

include
^^^^^^^

``{\% include FILENAME \%}``

Global parameters
-----------------

+----------------------------------------+--------------------------------------------------------------------------------------+
| Option [default value]                 | Description                                                                          |
+========================================+======================================================================================+
| ``DefaultModel`` [``Transcode``]       | Default layers model. Can be ``Frame``, ``Frame_CUDA``, ``Transcode`` or ``Spike``   |
+----------------------------------------+--------------------------------------------------------------------------------------+
| ``DefaultDataType`` [``Float32``]      | Default layers data type. Can be ``Float16``, ``Float32`` or ``Float64``             |
+----------------------------------------+--------------------------------------------------------------------------------------+
| ``SignalsDiscretization`` [0]          | Number of levels for signal discretization                                           |
+----------------------------------------+--------------------------------------------------------------------------------------+
| ``FreeParametersDiscretization`` [0]   | Number of levels for weights discretization                                          |
+----------------------------------------+--------------------------------------------------------------------------------------+
