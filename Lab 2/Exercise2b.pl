older(prince_charles,princess_ann).
older(princess_ann,prince_andrew).
older(prince_andrew,prince_edward).

older(X,Y) :-
    \+ X = Y,
    older(X,Z),
    older(Z,Y).

succession([X,Y|List]) :-
    older(X,Y),succession([Y|List]).
succession([X,Y]) :-
    older(X,Y),succession(Y).
succession([X]).

count([],N) :-
    N=0.
count([X|T],N):-
    count(T,N1),
    N is N1 + 1.

/* Call the method below to get the answer
royal(X,4) where 4 is the number of family members to consider*/
royal(X,N):-
    succession(X),
    count(X,N).