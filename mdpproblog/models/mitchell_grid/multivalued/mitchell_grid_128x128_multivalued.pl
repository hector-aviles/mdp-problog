% GRID 128 X 128
% StateSpace: 16384


% ---SDD ---
% Nodos: 5139
% Tiempo Grounding: 
% Tiempo Compilación: 

% Tiempo Value Iteration: 9275.8749
% Iteraciones: 5

% --- d-DDNNF ---
% Nodos: 109905
% Tiempo Grounding: 
% Tiempo Compilación: 

% Tiempo Value Iteration: 
% Iteraciones: TIMEOUT

% --- d-DDNNF + Darwiche ---
% Nodos: 112758
% Tiempo Grounding: 
% Tiempo Compilación: 

% Tiempo Value Iteration: 28388.3807
% Iteraciones: 5


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

row(1). row(2). row(3). row(4). row(5). row(6). row(7). row(8). row(9). row(10). row(11). row(12). row(13). row(14). row(15). row(16).
row(17). row(18). row(19). row(20). row(21). row(22). row(23). row(24). row(25). row(26). row(27). row(28). row(29). row(30). row(31). row(32).
row(33). row(34). row(35). row(36). row(37). row(38). row(39). row(40). row(41). row(42). row(43). row(44). row(45). row(46). row(47). row(48).
row(49). row(50). row(51). row(52). row(53). row(54). row(55). row(56). row(57). row(58). row(59). row(60). row(61). row(62). row(63). row(64).
row(65). row(66). row(67). row(68). row(69). row(70). row(71). row(72). row(73). row(74). row(75). row(76). row(77). row(78). row(79). row(80).
row(81). row(82). row(83). row(84). row(85). row(86). row(87). row(88). row(89). row(90). row(91). row(92). row(93). row(94). row(95). row(96).
row(97). row(98). row(99). row(100). row(101). row(102). row(103). row(104). row(105). row(106). row(107). row(108). row(109). row(110). row(111). row(112).
row(113). row(114). row(115). row(116). row(117). row(118). row(119). row(120). row(121). row(122). row(123). row(124). row(125). row(126). row(127). row(128).

col(1). col(2). col(3). col(4). col(5). col(6). col(7). col(8). col(9). col(10). col(11). col(12). col(13). col(14). col(15). col(16).
col(17). col(18). col(19). col(20). col(21). col(22). col(23). col(24). col(25). col(26). col(27). col(28). col(29). col(30). col(31). col(32).
col(33). col(34). col(35). col(36). col(37). col(38). col(39). col(40). col(41). col(42). col(43). col(44). col(45). col(46). col(47). col(48).
col(49). col(50). col(51). col(52). col(53). col(54). col(55). col(56). col(57). col(58). col(59). col(60). col(61). col(62). col(63). col(64).
col(65). col(66). col(67). col(68). col(69). col(70). col(71). col(72). col(73). col(74). col(75). col(76). col(77). col(78). col(79). col(80).
col(81). col(82). col(83). col(84). col(85). col(86). col(87). col(88). col(89). col(90). col(91). col(92). col(93). col(94). col(95). col(96).
col(97). col(98). col(99). col(100). col(101). col(102). col(103). col(104). col(105). col(106). col(107). col(108). col(109). col(110). col(111). col(112).
col(113). col(114). col(115). col(116). col(117). col(118). col(119). col(120). col(121). col(122). col(123). col(124). col(125). col(126). col(127). col(128).