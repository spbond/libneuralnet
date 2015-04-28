/*
 * ABSearch.cpp
 *
 *  Created on: Jun 20, 2014
 *      Author: shaun
 */

#include "ABSearch.h"

#include <limits>
using std::numeric_limits;

#include <algorithm>
using std::min;
using std::max;


namespace
{
const float MAX_FLOAT =  numeric_limits<float>::max();
const float MIN_FLOAT = -numeric_limits<float>::max();
}

namespace NN
{

// Constructor
// Pre: nn must be a valid neural net which will evaluate inputs provided by mf
//      mf must provide a vector of all possible moves for the game
//      player1 indicates whether or not the search
//          tree will evaluate the inputs as player 1
// Post: The AB_Search object will evaluate inputs using nn
//       The AB_Search tree will search to a max depth of md
//
AB_Search::AB_Search(shared_ptr<NeuralNet> nn, shared_ptr<MoveFinder> mf,
                     size_t md, bool player1)
: m_neuralnet(nn), m_finder(mf), m_max_depth(md), m_player(player1)
{}

AB_Search::~AB_Search()
{}

// board must have 32 values
// depth is >= 1
float
AB_Search::maximize( const vector<float>& board, float beta, size_t depth )
{
    if( depth < m_max_depth )
    {
        vector<vector<float>> moves = move(m_finder->possibleMoves(board, m_player));
        float alpha = MIN_FLOAT;

        for( auto ii = 0u; ii < moves.size(); ++ii )
        {
            alpha = max( alpha, minimize( moves[ii], alpha, depth + 1) );
            if( alpha >= beta )
            {
                break;
            }
        }

        return alpha;
    }
    else    //terminal node
    {
        return m_neuralnet->evaluate( board );
    }
}

// board must have 32 values
// depth is >= 1
float
AB_Search::minimize( const vector<float>& board, float beta, size_t depth )
{
    if( depth < m_max_depth )
    {
        vector<vector<float>> moves = move(m_finder->possibleMoves(board, !m_player));
        float alpha = MAX_FLOAT;

        for( auto ii = 0u; ii < moves.size(); ++ii )
        {
            alpha = min( alpha, maximize( moves[ii], alpha, depth + 1) );
            if( alpha <= beta )
            {
                break;
            }
        }

        return alpha;
    }
    else    // terminal node
    {
        return m_neuralnet->evaluate( board );
    }
}


vector<float>
AB_Search::nextMove(const vector<float>& current_state)
{
    vector<vector<float>> moves = move(m_finder->possibleMoves(current_state, m_player));
    vector<float> best_move = current_state;
    float alpha = MIN_FLOAT;

    // Find which possible move has the highest guaranteed value
    for( auto ii = 0u; ii < moves.size(); ++ii )
    {
        float value = minimize( moves[ii], alpha, 1 );
        if( value >= alpha )    // If it is better than the previous best, save it
        {
            alpha = value;
            best_move = move(moves[ii]);
        }
    }

    return best_move;
}

vector<float>
AB_Search::nextMove(const vector<float>& current_state, size_t max_depth)
{
    size_t default_depth = m_max_depth;
    m_max_depth = max_depth;

    vector<float> move = nextMove(current_state);

    m_max_depth = default_depth;

    return move;
}

vector<float>
AB_Search::operator ()(const vector<float>& current_state)
{
    return nextMove(current_state);
}

vector<float>
AB_Search::operator ()(const vector<float>& current_state, size_t max_depth)
{
    return nextMove(current_state, max_depth);
}

} /* namespace NN */
