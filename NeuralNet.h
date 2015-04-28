/*
 * NeuralNet.h
 *
 *  Created on: Jun 5, 2014
 *      Author: shaun
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <ctime>
#include <iostream>
using std::cout;
using std::endl;
#include <vector>
using std::vector;
#include <algorithm>
using std::swap;
#include <memory>
using std::shared_ptr;
#include "Synapse.h"


namespace NN
{

class NeuralNet
{
public:
    NeuralNet(const vector<size_t>& node_structure, const shared_ptr<Synapse>);
    NeuralNet(const NeuralNet&);

    virtual
    ~NeuralNet();

    virtual vector<size_t>
    structure() const;

    virtual float
    evaluate(const vector<float>& inputs);

protected:
    vector<vector<float>>   m_nodes;
    vector<float>           m_weights;
    shared_ptr<Synapse>     m_synapse;
};



class CheckersNet : public NeuralNet
{
public:
    enum Player {BLACK, RED};

    CheckersNet(const vector<size_t>& node_structure, shared_ptr<Synapse>);
    CheckersNet(CheckersNet&);

    virtual float
    evaluate(const vector<float>& inputs);

    virtual void
    player(Player color);

private:
    float m_king;
    Player m_player;
    float m_MAX;
    float m_MIN;
};

} // namespace NN

#endif /* NEURALNET_H_ */
