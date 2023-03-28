/* company declared */
company(sumsum).
company(appy).

/*sumsum is a competitor of appy*/
competitor_of(sumsum,appy).
competitor_of(X,Y):-
    competitor_of(Y,X).

/*stevey is the boss of appy*/
boss_of(stevey,appy).

/*X is a business, if X is a smart phone tech*/
business(X):-
    smart_phone_technology(X).

/*galactica-s3 is a smart phone tech*/
smart_phone_technology(galactica-s3).

/*stevey stole galactica from sumsum*/
steal(stevey,galactica-s3,sumsum).

rival(X,Y):-
    company(X),
    company(Y),
    competitor_of(X,Y).

/*A person is unethical if they are a boss of some company,
  and steals business from other company who is a rival*/
unethical(Boss):-
    steal(Boss,Business,Other),
    boss_of(Boss,Company),
    rival(Company,Other),
    business(Business).
