/*
    (C) Copyright 2011 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_GRAPHVIZ_H
#define N2D2_GRAPHVIZ_H

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <map>

#include "utils/Utils.hpp"

namespace N2D2 {
class GraphViz {
public:
    GraphViz(const std::string& name,
             const std::string& label = "",
             bool directed = false);
    void node(const std::string& name, const std::string& label = "");
    void edge(const std::string& start, const std::string& end,
              const std::string& label = "");
    void edges(const std::string& start, const std::vector<std::string>& end,
               const std::vector<std::string>& label
                = std::vector<std::string>());
    void edges(const std::vector<std::string>& start, const std::string& end,
               const std::vector<std::string>& label
                = std::vector<std::string>());
    void attr(const std::string& name,
              const std::string& attr,
              const std::string& value);
    void subgraph(const GraphViz& graph);
    void render(const std::string& fileName) const;
    virtual ~GraphViz();

protected:
    std::string escape(const std::string& str) const;

private:
    std::string mName;
    bool mDirected;
    unsigned int mSubIndex;
    std::map<std::string, std::vector<std::string> > mNodes;
    std::vector<std::pair<std::string, std::string> > mEdges;
    std::map<std::string,
        std::vector<std::pair<std::string, std::string> > > mAttrs;
};
}

#endif // N2D2_GRAPHVIZ_H
