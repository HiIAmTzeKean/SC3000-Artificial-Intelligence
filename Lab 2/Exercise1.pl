company(sumsum).
company(appy).
competitor_of(sumsum,appy).
boss_of(stevey,appy).

business(X):-smart_phone_technology(X). /*phone is business*/
smart_phone_technology(galatica_s3). /*Given in text*/

steal(stevey,galatica_s3,sumsum).
competitor_of(X,Y):-competitor_of(Y,X).
rival(X,Y):-competitor_of(X,Y).

unethical(Boss):-steal(Boss,Business,Other),boss_of(Boss,Company),rival(Company,Other),business(Business).
