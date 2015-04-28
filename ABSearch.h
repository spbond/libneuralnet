/*
 * ABSearch.h
 *
 *  Created on: Jun 20, 2014
 *      Author: shaun
 */

#ifndef ABSEARCH_H_
#define ABSEARCH_H_

#include "NeuralNet.h"
using NN::NeuralNet;
#include "MoveFinder.h"
using NN::MoveFinder;
#include <vector>
using std::vector;
#include <memory>
using std::shared_ptr;

namespace NN
{

class AB_Search
{
public:
    AB_Search(shared_ptr<NeuralNet>, shared_ptr<MoveFinder>,
              size_t max_depth, bool player1);

    virtual ~AB_Search();

    vector<float>
    nextMove(const vector<float>& current_state);

    vector<float>
    nextMove(const vector<float>& current_state, size_t max_depth);

    vector<float>
    operator()(const vector<float>& current_state);

    vector<float>
    operator()(const vector<float>& current_state, size_t max_depth);

private:
    shared_ptr<NeuralNet> m_neuralnet;
    shared_ptr<MoveFinder> m_finder;
    size_t m_max_depth;
    bool m_player;

    float
    minimize( const vector<float>& board, float beta, size_t depth );

    float
    maximize( const vector<float>& board, float beta, size_t depth );
};

} /* namespace NN */
#endif /* ABSEARCH_H_ */
