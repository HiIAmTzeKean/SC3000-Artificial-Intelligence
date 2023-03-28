male(prince_charles).
male(prince_andrew).
male(prince_edward).
female(princess_ann).

older(prince_charles,princess_ann).
older(princess_ann,prince_andrew).
older(prince_andrew,prince_edward).

older(X,Y) :-
    \+ X = Y,
    older(X,Z),
    older(Z,Y).

correct_order(X,Y) :-
    male(X),female(Y).
correct_order(X,Y) :-
    male(X),male(Y),older(X,Y).
correct_order(X,Y) :-
    female(X),female(Y),older(X,Y).

succession([X,Y|List]) :-
    correct_order(X,Y),succession([Y|List]).
succession([X,Y]) :-
    correct_order(X,Y),succession(Y).
succession([_]).

count([],N) :-
    N=0.
count([X|T],N):-
    count(T,N1),
    N is N1 + 1.

royal(X,N):-
    succession(X),
    count(X,N).