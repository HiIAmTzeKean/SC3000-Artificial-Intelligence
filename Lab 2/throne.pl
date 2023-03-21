male(prince_charles).
male(prince_andrew).
male(prince_edward).
female(princess_ann).

birth_first(queen_elizabeth,prince_charles).
birth_first(prince_charles,princess_ann).
birth_first(princess_ann,prince_andrew).
birth_first(prince_andrew,prince_edward).
monarch(queen_elizabeth,united_kingdom).

birth_first(X,Y) :-
    \+ X = Y,
    birth_first(X,Z),
    birth_first(Z,Y).

correct_order(X,Y) :-
    male(X),female(Y).
correct_order(X,Y) :-
    male(X), male(Y),
    birth_first(X,Y).
correct_order(X,Y) :-
    female(X), female(Y),
    birth_first(X,Y).

succession([X,Y|List]) :-
    correct_order(X,Y),succession([Y|List]).
succession([X,Y]) :-
    correct_order(X,Y),succession(Y).
succession(X) :-
    X = X.
