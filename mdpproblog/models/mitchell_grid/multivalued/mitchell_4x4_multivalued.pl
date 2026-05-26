% Mitchell Grid (4x4) — Factorized (multivalued AD) encoding
% Generated automatically by generate_benchmarks.py
% Fluents: 2 multivalued (x: base 4, y: base 4)
% State space: 16 (= 4 x 4), all valid
% Actions: 5
% Terminal: coor(1,4)

% State fluents
state_fluent(x(X), multivalued) :- row(X).
state_fluent(y(Y), multivalued) :- col(Y).

% Actions
action(left). action(right). action(up). action(down). action(stay).

% Reward
utility(goal, 100).
goal :- x(1, 0), y(3, 0), right.
goal :- x(2, 0), y(4, 0), up.

% Terminal (absorbing)
terminal :- x(1, 0), y(4, 0).

% Factor y (column) transitions
1.0::y(Y_new, 1) :- y(Y, 0), right, Y_new is Y + 1, col(Y_new), not(terminal).
1.0::y(Y, 1)     :- y(Y, 0), right, Y_new is Y + 1, not(col(Y_new)), not(terminal).

1.0::y(Y_new, 1) :- y(Y, 0), left, Y_new is Y - 1, col(Y_new), not(terminal).
1.0::y(Y, 1)     :- y(Y, 0), left, first_col(Y), not(terminal).

1.0::y(Y, 1)     :- y(Y, 0), up, not(terminal).
1.0::y(Y, 1)     :- y(Y, 0), down, not(terminal).
1.0::y(Y, 1)     :- y(Y, 0), stay, not(terminal).

% Factor x (row) transitions
1.0::x(X_new, 1) :- x(X, 0), down, X_new is X + 1, row(X_new), not(terminal).
1.0::x(X, 1)     :- x(X, 0), down, X_new is X + 1, not(row(X_new)), not(terminal).

1.0::x(X_new, 1) :- x(X, 0), up, X_new is X - 1, row(X_new), not(terminal).
1.0::x(X, 1)     :- x(X, 0), up, first_row(X), not(terminal).

1.0::x(X, 1)     :- x(X, 0), left, not(terminal).
1.0::x(X, 1)     :- x(X, 0), right, not(terminal).
1.0::x(X, 1)     :- x(X, 0), stay, not(terminal).

% Absorption for terminal states
1.0::x(X, 1) :- x(X, 0), terminal.
1.0::y(Y, 1) :- y(Y, 0), terminal.

% Helpers for wall bounce
first_col(1).
first_row(1).

% Background knowledge
row(1). row(2). row(3). row(4).
col(1). col(2). col(3). col(4).