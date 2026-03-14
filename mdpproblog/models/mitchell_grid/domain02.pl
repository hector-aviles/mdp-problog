% domain for 1x2, 2x2, 2x4, 4x4, 4x8, 8x8, 16x16 row-colum grids
% n states  [ 2,   4,   8,  16,  32,  128,  256 ] 

%state fluents
state_fluent(coor(X,Y), multivalued) :- row(X), col(Y).

%actions
action(left).
action(right).
action(up).
action(down).
action(stay).

%utilities
utility(goal, 100).

%reward model
goal :- coor(1,2, 1), not(terminal).

%restriction
terminal :- coor(1,2, 0).

%Transitions 

%valid horizontal moves
1.00::coor(X, Y_new, 1) :- coor(X, Y, 0), right, Y_new is Y + 1, col(Y_new), not(terminal).
1.00::coor(X, Y_new, 1) :- coor(X, Y, 0), left, Y_new is Y - 1, col(Y_new), not(terminal).

%valid vertical moves
1.00::coor(X_new, Y, 1) :- coor(X, Y, 0), down, X_new is X + 1, row(X_new), not(terminal).
1.00::coor(X_new, Y, 1) :- coor(X, Y, 0), up, X_new is X - 1, row(X_new), not(terminal).

%stay action transition (generalized for the whole grid)
1.00::coor(X, Y, 1):- coor(X, Y, 0), stay, not(terminal).

%Inertia rules

%Horizontal bounce (if trying to move out of columns)
1.00::coor(X, Y, 1):- coor(X, Y, 0), right, Y_new is Y + 1, not(col(Y_new)), not(terminal).
1.00::coor(X, Y, 1):- coor(X, Y, 0), left, Y_new is Y - 1, not(col(Y_new)), not(terminal).

% Vertical bounce (if trying to move out of rows)
1.00::coor(X, Y, 1):- coor(X, Y, 0), down, X_new is X + 1, not(row(X_new)), not(terminal).
1.00::coor(X, Y, 1):- coor(X, Y, 0), up, X_new is X - 1, not(row(X_new)), not(terminal).

% Any action in a terminal state returns to the same state
1.00::coor(X, Y, 1):- coor(X, Y, 0), terminal.
