/*
 * NeuralNet.cpp
 *
 *  Created on: Jun 5, 2014
 *      Author: shaun
 */

#include "NeuralNet.h"
#include <algorithm>
using std::swap;
#include <random>
using std::mt19937;
using std::uniform_real_distribution;


namespace NN
{

// Constructor for a new unique neural network.
// Pre:  number_of_nodes is the number of nodes at each layer including input.
//       Last entry in number_of_nodes must have size of 1.
//       synapse must be a valid Synapse.
// Post: m_nodes will have size equal to number_of_nodes.size().
//       Each element of m_nodes will contain a number of nodes equal to
//          the corresponding element of number_of_nodes.
//       For every node in a given layer, There will be one connection with
//          each node in the previous layer.
//       m_weights will be contain a random weight for each node connection.
NeuralNet::NeuralNet(const vector<size_t>& node_structure, const shared_ptr<Synapse> synapse)
: m_synapse(synapse)
{
    // Determine the number of weights needed and populate m_nodes.
    size_t total_weights = 0;
    m_nodes.push_back(vector<float>(node_structure[0]));
    for(size_t ii = 1; ii < node_structure.size(); ++ii)
    {
        // Each node is connected to each node in the previous layer.
        total_weights += node_structure[ii] * node_structure[ii-1];

        // Add a layer of nodes with the number of nodes in node_structure.
        m_nodes.push_back(vector<float>(node_structure[ii]));
    }

    // Store one weight for every weight needed.
    mt19937 rand_num;
    rand_num.seed(rand());
    uniform_real_distribution<float> uniform(-2.0, 2.0);
    m_weights.reserve(total_weights);
    for(size_t ii = 0; ii < total_weights; ++ii)
    {
        m_weights.push_back(uniform(rand_num));
    }
}

// Constructor which duplicates an existing neural network.
// Pre:  copy is a valid NeuralNet.
// Post: This NeuralNet will be an exact copy of copy.
NeuralNet::NeuralNet(const NeuralNet& copy)
: m_nodes(copy.m_nodes),
  m_weights(copy.m_weights),
  m_synapse(copy.m_synapse)
{}


NeuralNet::~NeuralNet() {
}

// Used to query the internal structure of the neural net.
// Pre:  None
// Post: Returns a vector containing the number of nodes in each layer.
vector<size_t>
NeuralNet::structure() const
{
    vector<size_t> node_structure;

    for(auto ii = 0u; ii < m_nodes.size(); ++ii)
    {
        node_structure.push_back(m_nodes[ii].size());
    }

    return node_structure;
}

// Evaluates the current set of input values to determine how "good" they are.
// Pre:  inputs is of size equal to the first layer of nodes.
// Post: Returns a floating point value.
float
NeuralNet::evaluate(const vector<float>& inputs)
{
    auto& synapse = *m_synapse;
    size_t current_weight = 0;

    // First time through, use inputs
    auto& out_nodes = m_nodes[1];
    out_nodes.assign(out_nodes.size(), 0.0);
    for(auto out_node = 0u; out_node < out_nodes.size(); ++out_node)
    {
        for(auto in_node = 0u; in_node < inputs.size(); ++in_node)
        {
            out_nodes[out_node] += inputs[in_node] * m_weights[current_weight];
            ++current_weight;
        }

        out_nodes[out_node] = synapse(out_nodes[out_node]);
    }

    // Thereafter, use previously written nodes
    for(auto layer = 1u; layer < m_nodes.size()-1; ++layer)
    {
        auto& out_nodes = m_nodes[layer+1];
        auto& in_nodes  = m_nodes[layer];
        out_nodes.assign(out_nodes.size(), 0.0);
        for(auto out_node = 0u; out_node < out_nodes.size(); ++out_node)
        {
            for(auto in_node = 0u; in_node < in_nodes.size(); ++in_node)
            {
                out_nodes[out_node] += in_nodes[in_node] * m_weights[current_weight];
                ++current_weight;
            }

            out_nodes[out_node] = synapse(out_nodes[out_node]);
        }
    }

    return m_nodes.back()[0];
}


// Constructor for a new unique checkers-based neural network.
// Pre:  number_of_nodes is the number of nodes at each layer including input.
//       Last entry in number_of_nodes must have size of 1.
// Post: m_nodes will have size equal to number_of_nodes.size().
//       Each element of m_nodes will contain a number of nodes equal to
//          the corresponding element of number_of_nodes.
//       For every node in a given layer, There will be one connection with
//          each node in the previous layer.
//       m_weights will be contain a random weight for each node connection.
//       m_king will have a value of 1.3.
CheckersNet::CheckersNet(const vector<size_t>& node_structure, shared_ptr<Synapse> synapse)
: NeuralNet(node_structure, synapse), m_king(1.3), m_player(BLACK)
{
    m_MAX = (*m_synapse)(std::numeric_limits<float>::max());
    m_MIN = (*m_synapse)(-std::numeric_limits<float>::max());
}

// Constructor which duplicates an existing neural network.
// Pre:  copy is a valid CheckersNet.
// Post: This CheckersNet will be an exact copy of copy.
CheckersNet::CheckersNet(CheckersNet& copy)
: NeuralNet(copy), m_king(copy.m_king), m_player(copy.m_player),
  m_MAX(copy.m_MAX), m_MIN(copy.m_MIN)
{}


// Evaluates the current set of input values to determine how "good" they are.
// Pre:  inputs is of size equal to the first layer of nodes.
// Post: Returns a floating point value.
float
CheckersNet::evaluate(const vector<float>& inputs)
{
    for(auto ii = 0u; ii < inputs.size(); ++ii)
    {
        if(inputs[ii] > 1.0)
        {
            m_nodes[0][ii] = m_king;
        }
        else if(inputs[ii] < -1.0)
        {
            m_nodes[0][ii] = -m_king;
        }
        else m_nodes[0][ii] = inputs[ii];
    }

    if( m_player == BLACK )
    {
        return NeuralNet::evaluate(m_nodes[0]);
    }
    else
    {
        return (m_MAX - NeuralNet::evaluate(m_nodes[0])) + m_MIN;
    }
}

void
CheckersNet::player(Player color)
{
    m_player = color;
}

} // End namespace NN
