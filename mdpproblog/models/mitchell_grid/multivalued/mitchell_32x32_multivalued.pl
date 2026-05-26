% Mitchell Grid (32x32) — Factorized (multivalued AD) encoding
% Generated automatically by generate_benchmarks.py
% Fluents: 2 multivalued (x: base 32, y: base 32)
% State space: 1024 (= 32 x 32), all valid
% Actions: 5
% Terminal: coor(1,32)

% State fluents
state_fluent(x(X), multivalued) :- row(X).
state_fluent(y(Y), multivalued) :- col(Y).

% Actions
action(left). action(right). action(up). action(down). action(stay).

% Reward
utility(goal, 100).
goal :- x(1, 0), y(31, 0), right.
goal :- x(2, 0), y(32, 0), up.

% Terminal (absorbing)
terminal :- x(1, 0), y(32, 0).

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
row(1). row(2). row(3). row(4). row(5). row(6). row(7). row(8). row(9). row(10). row(11). row(12). row(13). row(14). row(15). row(16). row(17). row(18). row(19). row(20). row(21). row(22). row(23). row(24). row(25). row(26). row(27). row(28). row(29). row(30). row(31). row(32).
col(1). col(2). col(3). col(4). col(5). col(6). col(7). col(8). col(9). col(10). col(11). col(12). col(13). col(14). col(15). col(16). col(17). col(18). col(19). col(20). col(21). col(22). col(23). col(24). col(25). col(26). col(27). col(28). col(29). col(30). col(31). col(32).