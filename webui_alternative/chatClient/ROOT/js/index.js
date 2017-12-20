$(function() {
	$("#inputArea").keyup(function(event) {
	  	var keyEvent=event||window.event; 
	  	switch(keyEvent.keyCode) {
	  	   	case 13: 
				submitQuestion();
				break; 
	  	}
	});
	
	$("#sendButton").click(function() {
		submitQuestion();
	});
});

function getTagContent(data, tag) {
    var i1=data.indexOf("<"+tag+">");
    var i2=data.indexOf("</"+tag+">");
    if ((i1>=0)	&& (i2>i1)) return data.substring(i1+tag.length+2,i2);
    return "";
}

function findNoError(data, redirect) {
    if (data.indexOf("<error>") >= 0) {
    	alert(getTagContent(data, "error"));
        return false;
    }
    return true;        	
}

function submitQuestion() {
	var question=$.trim($("#inputArea").val());
	if (question=="") {
		alert("Please type something before clicking the Send button.");
		return false;
	}
	var bc=$("#boxCenter");
	bc.append("<div class='question'><div class='user'>ME</div><div class='content'>"+question+"</div></div>");
	$("#inputArea").val("");
	
	var dataUrl="question="+encodeURIComponent(question);
	$.ajax({
		type:"POST", 
		url:"/ajax/getChatReply.jsp",
		data:dataUrl,
		error:function(){
			alert("Unexpected error happened!");
		},
		success:function(data){
			if (findNoError(data, 1)) {
				var reply=getTagContent(data, "reply");
				bc.append("<div class='answer'><div class='user'><figure class='avatar'><img src='/images/papaya.jpg'/></figure></div><div class='content'>"+reply+"</div></div>");
				bc.scrollTop(bc[0].scrollHeight);
			}
		}
	});
}