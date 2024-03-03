# Grievance_Redressal_System
It is a web application for classifying different grievances into specific domains using NLP 


Develop a web application named "Grievance Redressal System".

At the outset, the system is intended to obtain different grievances from the victims in our society and classify them into different crime categories.

The system uses a LSTM model built to determine the crime category of complaint received from the victims.

We have a database which has a prestored complaints 
of different cases such as robbery, assault, cyber_crime, murder etc.
This database is used for training the model for using it to further run the algorithm. We may assume here that, the accuracy of the model is directly proportional to the quantum/no of records in database. As huge database increases the learning curve of our model.

The web application framework used in the project is "FLASK". This framework is lightweight, has lot of libraries which would be ideal for us.

There are many functionalities in the project such as the following :

1) personal details of victim (Name, age, Police limits).

2) Acknowledgement generation with related legal proposition( law sections printed).


There are many advantages of tge model developed :

1) It ensures there need nit be any physical cintact with the legal or competent legal authority, so there is more space for personal liberty and nk influence of authorities involved.

2) Lowers the burden of officials as traditional paper work is not involved.

3) Individuals who are not educated or from non legal domain can understand the sections involved in their respective case which improves transparency.

The other end of the web application involves admin dashboard with features of sending an acknowledgement to the victims via mail.

Texhnology stack:: python libraries (pandas/Numpy/Matplotlib), Javascript, Html, css, Flask, sqlite3 ).
