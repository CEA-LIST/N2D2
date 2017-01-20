/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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
*/

#include "utils/TemplateParser.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(TemplateParser,
             render,
             (std::string templ, std::string render),
             std::make_tuple(
                 // Template
                 "{{b}}",
                 // Expected result
                 "12"),
             std::make_tuple(
                 // Template
                 "{% for a in range(3) %}"
                 "Essai {{a}}\n"
                 "{% endfor %}",
                 // Expected result
                 "Essai 0\n"
                 "Essai 1\n"
                 "Essai 2\n"),
             std::make_tuple(
                 // Template
                 "{% for a in range(1,3) %}"
                 "Essai {{a}}\n"
                 "{% endfor %}",
                 // Expected result
                 "Essai 1\n"
                 "Essai 2\n"),
             std::make_tuple(
                 // Template
                 "{% for a in range(c) %}"
                 "Essai {{a}}\n"
                 "{% endfor %}",
                 // Expected result
                 "Essai 0\n"
                 "Essai 1\n"
                 "Essai 2\n"
                 "Essai 3\n"),
             std::make_tuple(
                 // Template
                 "{% for a in range(1,c) %}"
                 "Essai {{a}}\n"
                 "{% endfor %}",
                 // Expected result
                 "Essai 1\n"
                 "Essai 2\n"
                 "Essai 3\n"),
             std::make_tuple(
                 // Template
                 "{% for a in range(c,d) %}"
                 "Essai {{a}}\n"
                 "{% endfor %}",
                 // Expected result
                 "Essai 4\n"),
             std::make_tuple(
                 // Template
                 "{% for a in range(d,7) %}"
                 "Essai {{a}}\n"
                 "{% endfor %}",
                 // Expected result
                 "Essai 5\n"
                 "Essai 6\n"),
             std::make_tuple(
                 // Template
                 "{% block items %}"
                 "  Title: {{.title}}\n"
                 "  Desc: {{.desc}}\n"
                 "{% endblock %}",
                 // Expected result
                 "  Title: title0\n"
                 "  Desc: desc0\n"
                 "  Title: title1\n"
                 "  Desc: desc1\n"))
{
    TemplateParser parser;
    parser.addParameter("b", 12);
    parser.addParameter("c", 4);
    parser.addParameter("d", 5);
    parser.addParameter("items", 2);
    parser.addParameter("items[0].title", "title0");
    parser.addParameter("items[0].desc", "desc0");
    parser.addParameter("items[1].title", "title1");
    parser.addParameter("items[1].desc", "desc1");

    std::stringstream result;
    parser.render(result, templ);

    ASSERT_EQUALS(result.str(), render);
}

RUN_TESTS()
