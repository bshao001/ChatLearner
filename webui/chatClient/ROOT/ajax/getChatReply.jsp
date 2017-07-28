<%@ page language="java" session="true" %>
<%@ page import="net.papayachat.chatservice.*" %>
<%
    response.setContentType("text/html; charset=utf-8");
    response.setHeader("Cache-Control", "no-cache");

	try {
		String question = request.getParameter("question");
		
		ChatClient cc = new ChatClient("http://papayachat.net:8080/ChatService", 5);
		String reply = cc.getReply(question).replaceAll("_np_", "<br/><br/>");
		
		response.getWriter().write("<rs><reply>" + reply + "</reply></rs>");
	} catch (Exception e) {
		e.printStackTrace();
		
		response.getWriter().write("<error>" + e.getMessage() + "</error>");
	}
%>