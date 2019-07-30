var $messages = $('.messages-content'),
    d, h, m,
    i = 0;

$(window).load(function() {
    $messages.mCustomScrollbar();
    setTimeout(function() {
        fakeMessage();
    }, 100);
});

function updateScrollbar() {
    $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
        scrollInertia: 10,
        timeout: 0
    });
}

function setDate() {
    d = new Date()
    if (m != d.getMinutes()) {
        m = d.getMinutes();
        $('<div class="timestamp">' + d.getHours() + ':' + m + '</div>').appendTo($('.message:last'));
    }
}

function insertMessage() {
    msg = $('.message-input').val();
    if ($.trim(msg) == '') {
        return false;
    }
    $('<div class="message message-personal">' + msg +'</div>').appendTo($('.mCSB_container')).addClass('new');
    setDate();
    $('.message-input').val(null);
    updateScrollbar();
    $('<div class="message loading new"><figure class="avatar"><img src="/static/img/Tb_logo.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
    updateScrollbar();
    setTimeout(function() {
		runPyScript(msg);
    },10);

}

 function runPyScript(input){
        updateScrollbar();
		var URL = "http://localhost:5000/chat?question="+input;
		console.log(URL);
		var res = 'Sorry I wasn\'t able to answer that';
		$.ajax({
	        async: true,  //og was false but it's been deprecated. switch if problems
	        type: 'GET',
	        url : URL,
	        success: function(data) {
	            $('.message.loading').remove();
		        console.log(data);
		        res = data;
		        $('<div class="message new"><figure class="avatar"><img src="/static/img/Tb_logo.png" /></figure>' + res + '</div>').appendTo($('.mCSB_container')).addClass('new');
                setDate();
                updateScrollbar();
	        }
        });
}


$('.message-submit').click(function() {
    insertMessage();
});

$(window).on('keydown', function(e) {
    if (e.which == 13) {
        insertMessage();
        return false;
    }
})

var Fake = [
    'Hi there, I\'m TriviaBot! Go ahead and ask a question',
    'Do you wanna know My Secret Identity?',
    'Nice to meet you',
    'How are you?',
    'Not too bad, thanks',
    'What do you do?',
    'That\'s awesome',
    'Codepen is a nice place to stay',
    'I think you\'re a nice person',
    'Why do you think that?',
    'Can you explain?',
    'Anyway I\'ve gotta go now',
    'It was a pleasure chat with you',
    'Bye',
    ':)'
]

function fakeMessage() {
    if ($('.message-input').val() != '') {
        return false;
    }
    $('<div class="message loading new"><figure class="avatar"><img src="/static/img/Tb_logo.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
    updateScrollbar();

    setTimeout(function() {
        $('.message.loading').remove();
        $('<div class="message new"><figure class="avatar"><img src="/static/img/Tb_logo.png" /></figure>' + Fake[i] + '</div>').appendTo($('.mCSB_container')).addClass('new');
        setDate();
        updateScrollbar();
        i++;
    }, 0);
}
