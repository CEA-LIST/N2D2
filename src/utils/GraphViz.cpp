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

#include "utils/GraphViz.hpp"

N2D2::GraphViz::GraphViz(const std::string& name,
                         const std::string& label,
                         bool directed):
    mName(name),
    mDirected(directed),
    mSubIndex(0)
{
    //ctor
    mNodes.insert(std::make_pair(name, std::vector<std::string>()));

    if (!label.empty())
        attr(name, "label", label);
}

void N2D2::GraphViz::node(const std::string& name,
                          const std::string& label)
{
    mNodes.find(mName)->second.push_back(name);

    if (!label.empty())
        attr(name, "label", label);
}

void N2D2::GraphViz::edge(const std::string& start,
                          const std::string& end,
                          const std::string& label)
{
    mEdges.push_back(std::make_pair(start, end));

    if (!label.empty())
        attr(start + "->" + end, "label", label);
}

void N2D2::GraphViz::edges(const std::string& start,
                           const std::vector<std::string>& end,
                           const std::vector<std::string>& labels)
{
    for (unsigned int i = 0, size = end.size(); i < size; ++i) {
        const std::string label = (i < labels.size()) ? labels[i] : "";
        edge(start, end[i], label);
    }
}

void N2D2::GraphViz::edges(const std::vector<std::string>& start,
                           const std::string& end,
                           const std::vector<std::string>& labels)
{
    for (unsigned int i = 0, size = start.size(); i < size; ++i) {
        const std::string label = (i < labels.size()) ? labels[i] : "";
        edge(start[i], end, label);
    }
}

void N2D2::GraphViz::attr(const std::string& name,
                          const std::string& attr,
                          const std::string& value)
{
    std::map<std::string,
        std::vector<std::pair<std::string, std::string> > >::iterator it;
    std::tie(it, std::ignore) = mAttrs.insert(std::make_pair(name,
                    std::vector<std::pair<std::string, std::string> >()));

    (*it).second.push_back(std::make_pair(attr, value));
}

void N2D2::GraphViz::subgraph(const GraphViz& graph)
{
    mEdges.insert(mEdges.end(), graph.mEdges.begin(), graph.mEdges.end());
    mAttrs.insert(graph.mAttrs.begin(), graph.mAttrs.end());

    for (std::map<std::string, std::vector<std::string> >::const_iterator it
         = graph.mNodes.begin(), itEnd = graph.mNodes.end(); it != itEnd; ++it)
    {
        std::stringstream graphName;
        graphName << (*it).first << mSubIndex;

        mNodes.insert(std::make_pair(graphName.str(), (*it).second));

        // Rename mAttrs key
        const std::map<std::string,
            std::vector<std::pair<std::string, std::string> > >
            ::const_iterator itGraphAttrs = mAttrs.find((*it).first);

        if (itGraphAttrs != mAttrs.end()) {
            mAttrs.insert(std::make_pair(graphName.str(),
                                         (*itGraphAttrs).second));
            mAttrs.erase((*it).first);
        }
    }

    ++mSubIndex;
}

void N2D2::GraphViz::render(const std::string& fileName) const
{
    std::ofstream dot(fileName.c_str());

    if (!dot.good())
        throw std::runtime_error("Could not open command file: " + fileName);

    if (mDirected)
        dot << "digraph " << escape(mName) << " {\n";
    else
        dot << "graph " << escape(mName) << " {\n";

    const std::map<std::string,
        std::vector<std::pair<std::string, std::string> > >
        ::const_iterator itGraphAttrs = mAttrs.find(mName);

    if (itGraphAttrs != mAttrs.end()) {
        for (std::vector<std::pair<std::string, std::string> >
             ::const_iterator itAttrs = (*itGraphAttrs).second.begin(),
             itAttrsEnd = (*itGraphAttrs).second.end();
             itAttrs != itAttrsEnd; ++itAttrs)
        {
            dot << "  " << (*itAttrs).first << "="
                << escape((*itAttrs).second) << ";\n";
        }
    }

    for (std::map<std::string, std::vector<std::string> >::const_iterator it
         = mNodes.begin(), itEnd = mNodes.end(); it != itEnd; ++it)
    {
        if ((*it).first != mName) {
            dot << "  subgraph " << escape((*it).first) << " {\n";

            const std::map<std::string,
                std::vector<std::pair<std::string, std::string> > >
                ::const_iterator itGraphAttrs = mAttrs.find((*it).first);

            if (itGraphAttrs != mAttrs.end()) {
                for (std::vector<std::pair<std::string, std::string> >
                     ::const_iterator itAttrs = (*itGraphAttrs).second.begin(),
                     itAttrsEnd = (*itGraphAttrs).second.end();
                     itAttrs != itAttrsEnd; ++itAttrs)
                {
                    dot << "    " << (*itAttrs).first << "="
                        << escape((*itAttrs).second) << ";\n";
                }
            }
        }

        for (std::vector<std::string>::const_iterator itNodes
             = (*it).second.begin(), itNodesEnd = (*it).second.end();
             itNodes != itNodesEnd; ++itNodes)
        {
            if ((*it).first != mName)
                dot << "  ";

            dot << "  " << escape(*itNodes);

            const std::map<std::string,
                std::vector<std::pair<std::string, std::string> > >
                ::const_iterator itNodeAttrs = mAttrs.find(*itNodes);

            if (itNodeAttrs != mAttrs.end()) {
                for (std::vector<std::pair<std::string, std::string> >
                     ::const_iterator itAttrs = (*itNodeAttrs).second.begin(),
                     itAttrsEnd = (*itNodeAttrs).second.end();
                     itAttrs != itAttrsEnd; ++itAttrs)
                {
                    dot << "[" << (*itAttrs).first << "="
                        << escape((*itAttrs).second) << "]";
                }
            }

            dot << ";\n";
        }

        if ((*it).first != mName)
            dot << "  }\n";
    }

    for (std::vector<std::pair<std::string, std::string> >::const_iterator it
         = mEdges.begin(), itEnd = mEdges.end(); it != itEnd; ++it)
    {
        dot << "  " << escape((*it).first) << " -> " << escape((*it).second);

        const std::string attrName = (*it).first + "->" + (*it).second;
        const std::map<std::string,
            std::vector<std::pair<std::string, std::string> > >
            ::const_iterator itEdgeAttrs = mAttrs.find(attrName);

        if (itEdgeAttrs != mAttrs.end()) {
            for (std::vector<std::pair<std::string, std::string> >
                 ::const_iterator itAttrs = (*itEdgeAttrs).second.begin(),
                 itAttrsEnd = (*itEdgeAttrs).second.end();
                 itAttrs != itAttrsEnd; ++itAttrs)
            {
                dot << "[" << (*itAttrs).first << "="
                    << escape((*itAttrs).second) << "]";
            }
        }

        dot << ";\n";
    }

    dot << "}\n";
    dot.close();

    const std::string cmd = "dot -Tpng " + fileName
        + " -o " + Utils::fileBaseName(fileName) + ".png";
    const int ret = system(cmd.c_str());

    if (ret < 0) {
        std::cout << Utils::cwarning << "Warning: could not plot GraphViz dot "
            "file " << fileName << " (return code: " << ret << ")"
            << Utils::cdef << std::endl;
    }
}

N2D2::GraphViz::~GraphViz()
{
    //dtor
}

std::string N2D2::GraphViz::escape(const std::string& str) const
{
    if (std::find_if(str.begin(), str.end(), Utils::isNotValidIdentifier)
        != str.end() || (!str.empty() && !isalpha(str[0])))
    {
        std::stringstream quotedStr;
        quotedStr << Utils::quoted(str);
        return quotedStr.str();
    }
    else
        return str;
}
