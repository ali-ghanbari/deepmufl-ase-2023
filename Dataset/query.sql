SELECT title,concat('https://stackoverflow.com/questions/',id), tags, score, creationDate From Posts
where body like '%<code>%' and
-- Limiting to Keras Only
tags like '%<keras>%' 
and  score >= 0 and
--Eliminating the posts facing syntax errors
(body NOT like '%IndexError%'
and body NOT like '%ValueError%'
and body NOT like '%TypeError%'
and body NOT like '%Traceback%')
and AnswerCount > 0 and
AcceptedAnswerId > 0 and
--Limiting to relavent keywords in posts
-- we also excluded tutorials, e.g., how to create custom loss function
(body like '%error%' 
or body like '%bug%'
or body like '%not work%' 
or body like '%fail%'
or body like '%accuracy%'
or body like '%expect%' 
or body like '%problem%' 
or body like '%fault%'
or body like '%fix%'
or body like '%issue%'
or body like '%loss%' 
or body like '%activation function%' 
or body like '%layer%' 
or body like '%last layer%' 
or body like '%hidden layer%' 
or body like '%bad performance%' 
or body like '%converge%' 
or body like '%not converge%' 
or body like '%nan%' 
or body like '%advanced activation%' 
or body like '%parameter mistakes%' 
or body like '%incorrect layer%' 
or body like '%inaccuracy%' 
or body like '%loss does not change%'
or body like '%weight does not change%'
or body like '%Model does not learn%' 
or body like '%wrong%'
or body like '%crash%'
or body like '%incorrect%'
or body like '%incompatible%'
or body like '%low accuracy%'
or body like '%invalid%'
or body like '%worse%'
or body like '%unexpected%'
or body like '%high loss%'
or body like '%misinterpret%'
or body like '%unknown%'
or body like '%dead relu%'
or body like '%poor weight initialization%'
or body like '%saturated activation%'
or body like '%vanishing gradient%'
or body like '%exploding gradient%'
or body like '%unchanged tensor%'
or body like '%zero%')
