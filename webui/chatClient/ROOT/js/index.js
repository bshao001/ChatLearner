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
				var jsCode=getTagContent(data, "jsCode");
				bc.append("<div class='answer'><div class='user'><figure class='avatar'><img src='/images/papaya.jpg'/></figure></div><div class='content'>"+reply+"</div></div>");
				bc.scrollTop(bc.prop("scrollHeight"));
				
				if (jsCode!="") {
				 	if (jsCode.indexOf("show_picture_randomly_para1_")==0) {
						var idx="show_picture_randomly_para1_".length;
						var pic_name=$.trim(jsCode.substring(idx));
						show_picture_randomly(bc, pic_name);
					}
				}
				
			}
		}
	});
}

function show_picture_randomly(bc, picture_name) {
	var lower_name=picture_name.toLowerCase();
	if (lower_name=="good_morning" || lower_name=="good_afternoon" || lower_name=="good_evening" || lower_name=="good_night") {
		var rand=Math.floor(Math.random()*6); // 5 images for each category
		if (rand>0) { // Do nothing when it is 0
			var img_src="/images/"+lower_name+"_"+rand+".jpg";
			var img_line="<div class='answer'>"+
				         "<div class='user'><figure class='avatar'><img src='/images/papaya.jpg'/></figure></div>"+
				         "<div class='image'><img src='"+img_src+"' alt='"+picture_name.replace("_", " ")+"'></div>"+
					     "</div>";
			bc.append(img_line);
			bc.stop().animate({scrollTop: bc.prop("scrollHeight")}, 800);
		}
	}
}