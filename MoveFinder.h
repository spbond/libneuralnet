/*
 * MoveFinder.h
 *
 *  Created on: Jun 20, 2014
 *      Author: shaun
 */

#ifndef MOVEFINDER_H_
#define MOVEFINDER_H_

#include <vector>
using std::vector;
using std::size_t;

namespace NN
{

class MoveFinder
{
public:
    MoveFinder();
    virtual ~MoveFinder();

    virtual const vector<vector<float>>&
    possibleMoves(const vector<float>& current_state, bool side1) = 0;
};


class CheckersMoveFinder : public MoveFinder
{
public:
    CheckersMoveFinder();

    virtual const vector<vector<float>>&
    possibleMoves(const vector<float>& current_state, bool black_player);

private:
    const float KING = 1.5;
    const float PAWN = 1.0;
    vector<vector<float>> m_moves;
    vector<float> m_state;
    int m_side;

    bool
    odd(unsigned) const;

    float&
    pieceAt(size_t row, size_t col);

    bool
    frontRightJump(size_t row, size_t col);

    bool
    frontLeftJump(size_t row, size_t col);

    bool
    backRightJump(size_t row, size_t col);

    bool
    backLeftJump(size_t row, size_t col);

    bool
    jump(size_t row, size_t col, float& neighbor, float& target, float& current);

    void
    findJump(size_t row, size_t col);

    bool
    tryToJump(float piece, size_t row, size_t col);

    void
    move(size_t row1, size_t col1, size_t row2, size_t col2);
};

} /* namespace NN */
#endif /* MOVEFINDER_H_ */
