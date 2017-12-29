<%@ page language="java" session="true" %>
<%@ page import="net.papayachat.chatservice.*" %>
<%
    response.setContentType("text/html; charset=utf-8");
    response.setHeader("Cache-Control", "no-cache");

	ChatClient cc = (ChatClient)session.getAttribute("ChatClient");
    if (cc == null) {
		cc = new ChatClient("http://papayachat.net:5000/reply", 5);
		session.setAttribute("ChatClient", cc);
	}
		
	try {
		String question = request.getParameter("question");
		String reply = cc.getReply(question).replaceAll("_np_", "<br/><br/>");
		
		// Tomcat will forward them to the log file. This is a dirty implementation. For a 
		// busy site, please cache them in memory (a synchronized list), and periodically 
		// dump the data in the list to a dedicated log file in a scheduled task.
		System.out.println("Q: " + question);
		System.out.println("A: " + reply);
		
		response.getWriter().write("<rs><reply>" + reply + "</reply></rs>");
	} catch (Exception e) {
		e.printStackTrace();
		
		response.getWriter().write("<error>" + e.getMessage() + "</error>");
	}
%>