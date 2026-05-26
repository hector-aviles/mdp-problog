% GRID 64 X 64
% StateSpace: 4096

% ---SDD ---
% Nodos: 2579
% Tiempo Grounding: 
% Tiempo Compilación: 

% Tiempo Value Iteration: 588.8841
% Iteraciones: 4

% --- d-DDNNF ---
% Nodos: 33373 | 34353
% Tiempo Grounding:  
% Tiempo Compilación: 

% Tiempo Value Iteration: TIMEOUT (8 horas)
% Iteraciones: 

% --- d-DDNNF + Darwiche ---
% Nodos: 33373 | 34353
% Tiempo Grounding: 1.7019
% Tiempo Compilación: 0.6399s

% Tiempo Value Iteration: 1894.5094s
% Iteraciones: 4

% State fluents
state_fluent(x(X), multivalued) :- row(X).
state_fluent(y(Y), multivalued) :- col(Y).

% Actions
action(left). action(right). action(up). action(down). action(stay).

% Reward
utility(goal, 100).
goal :- x(1, 0), y(2, 0), right.
goal :- x(2, 0), y(3, 0), up.

% Terminal state
terminal :- x(1, 0), y(3, 0).

% Factor y (columna)
1.0::y(Y_new, 1) :- y(Y, 0), right, Y_new is Y + 1, col(Y_new), not(terminal).
1.0::y(Y, 1)     :- y(Y, 0), right, Y_new is Y + 1, not(col(Y_new)), not(terminal).

1.0::y(Y_new, 1) :- y(Y, 0), left, Y_new is Y - 1, col(Y_new), not(terminal).
1.0::y(Y, 1)     :- y(Y, 0), left, first_col(Y), not(terminal).

1.0::y(Y, 1)     :- y(Y, 0), up, not(terminal).
1.0::y(Y, 1)     :- y(Y, 0), down, not(terminal).
1.0::y(Y, 1)     :- y(Y, 0), stay, not(terminal).

% Factor x (fila)
1.0::x(X_new, 1) :- x(X, 0), down, X_new is X + 1, row(X_new), not(terminal).
1.0::x(X, 1)     :- x(X, 0), down, X_new is X + 1, not(row(X_new)), not(terminal).

1.0::x(X_new, 1) :- x(X, 0), up, X_new is X - 1, row(X_new), not(terminal).
1.0::x(X, 1)     :- x(X, 0), up, first_row(X), not(terminal).

1.0::x(X, 1)     :- x(X, 0), left, not(terminal).
1.0::x(X, 1)     :- x(X, 0), right, not(terminal).
1.0::x(X, 1)     :- x(X, 0), stay, not(terminal).

% Absortion for terminal states
1.0::x(X, 1) :- x(X, 0), terminal.
1.0::y(Y, 1) :- y(Y, 0), terminal.

% Helpers for wall bounce
first_col(1).
first_row(1).

row(1). row(2). row(3). row(4). row(5). row(6). row(7). row(8).
row(9). row(10). row(11). row(12). row(13). row(14). row(15). row(16).
row(17). row(18). row(19). row(20). row(21). row(22). row(23). row(24).
row(25). row(26). row(27). row(28). row(29). row(30). row(31). row(32).
row(33). row(34). row(35). row(36). row(37). row(38). row(39). row(40).
row(41). row(42). row(43). row(44). row(45). row(46). row(47). row(48).
row(49). row(50). row(51). row(52). row(53). row(54). row(55). row(56).
row(57). row(58). row(59). row(60). row(61). row(62). row(63). row(64).

col(1). col(2). col(3). col(4). col(5). col(6). col(7). col(8).
col(9). col(10). col(11). col(12). col(13). col(14). col(15). col(16).
col(17). col(18). col(19). col(20). col(21). col(22). col(23). col(24).
col(25). col(26). col(27). col(28). col(29). col(30). col(31). col(32).
col(33). col(34). col(35). col(36). col(37). col(38). col(39). col(40).
col(41). col(42). col(43). col(44). col(45). col(46). col(47). col(48).
col(49). col(50). col(51). col(52). col(53). col(54). col(55). col(56).
col(57). col(58). col(59). col(60). col(61). col(62). col(63). col(64).

