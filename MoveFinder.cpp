/*
 * MoveFinder.cpp
 *
 *  Created on: Jun 20, 2014
 *      Author: shaun
 */

#include "MoveFinder.h"


namespace NN
{

MoveFinder::MoveFinder()
{}

MoveFinder::~MoveFinder()
{}


CheckersMoveFinder::CheckersMoveFinder()
: m_side(0)
{}

float&
CheckersMoveFinder::pieceAt(size_t row, size_t col)
{
    return m_state[row*4 + col];
}

bool
CheckersMoveFinder::frontRightJump(size_t row, size_t col)
{
    auto& neighbor_square = pieceAt(row + 1, col + 1 - row % 2);
    auto& target_square = pieceAt(row + 2, col + 1);
    auto& our_square = pieceAt(row, col);

    return jump(row + 2, col + 1, neighbor_square, target_square, our_square);
}

bool
CheckersMoveFinder::frontLeftJump(size_t row, size_t col)
{
    auto& neighbor_square = pieceAt(row + 1, col - row % 2);
    auto& target_square = pieceAt(row + 2, col - 1);
    auto& our_square = pieceAt(row, col);

    return jump(row + 2, col - 1, neighbor_square, target_square, our_square);
}

bool
CheckersMoveFinder::backRightJump(size_t row, size_t col)
{
    auto& neighbor_square = pieceAt(row - 1, col + 1 - row % 2);
    auto& target_square = pieceAt(row - 2, col + 1);
    auto& our_square = pieceAt(row, col);

    return jump(row - 2, col + 1, neighbor_square, target_square, our_square);
}

bool
CheckersMoveFinder::backLeftJump(size_t row, size_t col)
{
    auto& neighbor_square = pieceAt(row - 1, col - row % 2);
    auto& target_square = pieceAt(row - 2, col - 1);
    auto& our_square = pieceAt(row, col);

    return jump(row - 2, col - 1, neighbor_square, target_square, our_square);
}

bool
CheckersMoveFinder::jump(size_t row, size_t col, float& neighbor_square,
                         float& target_square, float& our_square)
{
    if( (neighbor_square * m_side < 0.0) && (target_square == 0.0) )
    {
        float captured_piece = neighbor_square;

        neighbor_square = 0.0;
        target_square = our_square;
        our_square = 0.0;

        if( ((row == 7) || (row == 0)) &&
            (target_square * m_side <= PAWN + .1) )
        { // turn it into a king
            target_square = m_side * KING;
            m_moves.push_back(m_state);
            target_square = m_side * PAWN;
        }
        else
        { // try to keep jumping
            findJump(row, col);
        }

        neighbor_square = captured_piece;
        our_square = target_square;
        target_square = 0.0;

        return true;
    }

    return false;
}

bool
CheckersMoveFinder::tryToJump(float piece, size_t row, size_t col)
{
    bool can_jump = false;

    if( (piece < -PAWN - .1 || piece > -PAWN + .1) && row < 6) // anything but a red pawn
    {
        if( col > 0 )
        {
            can_jump = frontLeftJump(row, col) || can_jump;
        }

        if( col < 3 )
        {
            can_jump = frontRightJump(row, col) || can_jump;
        }

    }
    if( (piece < PAWN - .1 || piece > PAWN + .1) && row > 1 ) // anything but a black pawn
    {
        if( col > 0 )
        {
            can_jump = backLeftJump(row, col) || can_jump;
        }

        if( col < 3 )
        {
            can_jump = backRightJump(row, col) || can_jump;
        }
    }

    return can_jump;
}

void
CheckersMoveFinder::findJump(size_t row, size_t col)
{
    bool can_jump = tryToJump(pieceAt(row, col), row, col);

    if( !can_jump )
    {
        m_moves.push_back(m_state);
    }
}

void
CheckersMoveFinder::move(size_t row1, size_t col1, size_t row2, size_t col2)
{
    auto& to = pieceAt(row2, col2);

    if( to < 0.1 && to > -0.1 )
    {
        auto& from = pieceAt(row1, col1);

        if( (row2 == 7 || row2 == 0) && (from * m_side <= PAWN + .1) )
        {
            to = m_side *KING;
            from = 0.0;

            m_moves.push_back(m_state);

            from = m_side * PAWN;
            to = 0.0;
        }
        else
        {
            to = from;
            from = 0.0;

            m_moves.push_back(m_state);

            from = to;
            to = 0.0;
        }
    }
}

const vector<vector<float>>&
CheckersMoveFinder::possibleMoves(const vector<float>& current_state, bool black_player)
{
    if(black_player)
        m_side = 1;
    else
        m_side = -1;

    m_state = current_state;
    m_moves.clear();
    bool can_jump = false;

    // Check for possible jumps first, since they must be taken
    for(auto row = 0u; row < 8u; ++row)
    {
        for(auto col = 0u; col < 4u; ++col)
        {
            auto piece = pieceAt(row, col);
            if(piece * m_side > 0.1) // our piece
            {
                can_jump = tryToJump(piece, row, col) || can_jump;
            }
        }
    }

    // If no jumps are possible, check for other possible moves
    if( !can_jump )
    {
        for(auto row = 0u; row < 8u; ++row)
        {
            for(auto col = 0u; col < 4u; ++col)
            {
                auto& piece = pieceAt(row, col);
                if( piece * m_side > 0.1 ) // our piece
                {
                    if( !odd(row) )
                    {
                        if( piece < -PAWN - .1 || piece > -PAWN + .1) // anything but red pawn
                        {
                            // front left
                            move(row, col, row+1, col);

                            if(col < 3)
                            {
                                // front right
                                move(row, col, row+1, col+1);
                            }
                        }
                        
                        if( (piece < PAWN -.1 || piece > PAWN + .1) && (row > 0) ) // anything but black pawn
                        {
                            // back left
                            move(row, col, row-1, col);

                            if(col < 3)
                            {
                                // back right
                                move(row, col, row-1, col+1);
                            }
                        }
                    }
                    else
                    {
                        if( (piece < -PAWN -.1 || piece > -PAWN + .1) && row < 7) // anything but a red pawn
                        {
                            // front right
                            move(row, col, row+1, col);

                            if(col > 0)
                            {
                                // front left
                                move(row, col, row+1, col-1);
                            }
                        }

                        if( piece < PAWN - .1 || piece > PAWN + .1 ) // anything but a black pawn
                        {
                            // back right
                            move(row, col, row-1, col);

                            if(col > 0)
                            {
                                // back left
                                move(row, col, row-1, col-1);
                            }
                        }
                    }
                }
            }
        }
    }

    return m_moves;
}

bool
CheckersMoveFinder::odd(unsigned integer) const
{
    return integer % 2;
}

} /* namespace NN */
