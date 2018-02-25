<%@ page language="java" session="true" %>
<%@ page import="net.papayachat.chatservice.*" %>
<%
    response.setContentType("text/html; charset=utf-8");
    response.setHeader("Cache-Control", "no-cache");
	
	ChatClient cc = (ChatClient)session.getAttribute("ChatClient");
    if (cc == null) {
		cc = new ChatClient("http://papayachat.net:8080/ChatService", 5);
		session.setAttribute("ChatClient", cc);
	}

	try {
		String question = request.getParameter("question");
		String reply = cc.getReply(question).replaceAll("_np_", "<br/><br/>").trim();
		
		// Tomcat will forward them to the log file. This is a dirty implementation. For a 
		// busy site, please cache them in memory (a synchronized list), and periodically 
		// dump the data in the list to a dedicated log file in a scheduled task.
		System.out.println("Q: " + question);
		System.out.println("A: " + reply);
		
		// Check if the last word is instructing client code. 
		// Add +1 so that it works even with a single word case.
		String lastWord = reply.substring(reply.lastIndexOf(" ")+1);
		String jsCode = "";
		if (lastWord.startsWith("_cc_start_") && lastWord.endsWith("_cc_end_")) {
			int idx1 = "_cc_start_".length();
			int idx2 = lastWord.lastIndexOf("_cc_end_");
			jsCode = lastWord.substring(idx1, idx2);
			
			int idx = reply.lastIndexOf(" _cc_start_");
			reply = reply.substring(0, idx).trim();
		}
		
		response.getWriter().write("<rs><reply>" + reply + "</reply><jsCode>" + jsCode + "</jsCode></rs>");
	} catch (Exception e) {
		e.printStackTrace();
		
		response.getWriter().write("<error>" + e.getMessage() + "</error>");
	}
%>