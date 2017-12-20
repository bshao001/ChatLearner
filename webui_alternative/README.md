# Web User Interface

If the SOAP-based web UI is not your preference, try this one, which is based on REST API using Flask. This alternative 
is also prepared as an example required for project: 
[TF-Model-Deploy-Tutorial](https://github.com/bshao001/TF-Model-Deploy-Tutorial).
 
Python is the number one programming language used for machine learning, while Java is still the most popular language 
for web development. The purpose of this web UI is also to demonstrate how to deploy a neural network model created and 
trained in Python and TensorFlow into a Java environment. A web service based on REST API is generated to meet this 
need. The steps to create the running environment in a Windows system are described below. Similar procedures can be 
followed in Linux, but have not been tried.

The description below assumes that you are configuring DNS entry papayachat.net to point to your machine, and running 
the web service server and client on the same one. In order to make the DNS entry papayachat.net work, edit the hosts 
file (located at C:\Windows\System32\drivers\etc in a normal installation) and add the following line:
    
    127.0.0.1  		papayachat.net

## Python Server

Flask (version 0.12.2) is employed to create the web service server. The installation is simple.

```bash
pip install flask
```

When flask is installed, run the commands below to bring up the web service. 

```bash
cd webui_alternative
cd server
python chatservice.py
```

## Java Client

The Java client is tested with Java 1.7 and Tomcat 7.0. You can try later versions if you prefer. Following the steps below to prepare the client:

1. Install Java 1.7.
2. Install Tomcat 7.0. Choose port 80 (at least not port 5000 if you are running the python web service on the same machine, which is on port 5000).
3. Copy the whole folder chatClient to C drive as C:\chatClient. 
4. Make changes to the server.xml of the Tomcat installation to add the chatClient web application (copy the Host portion for chatClient into server.xml). 
You need to change the name of the host if you are using a different domain.
5. Change the service address in C:\chatClient\ROOT\ajax\getChatReply.jsp in case you are using a different domain.
6. Restart Tomcat.

And now you are ready to try it: http://papayachat.net, if you are using this domain.
