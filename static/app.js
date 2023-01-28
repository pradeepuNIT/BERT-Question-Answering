
$(document).ready(function() {

	function set_alert(message, type='primary', id='') {
		var alert_msg = '<div class="alert alert-'+type+' text-center alert-dismissible" role="alert" id="alert-box ' + id + '">' +
				message
			"</div>";
	    $("#alerts").html(alert_msg);
	}

	function get_titles() {
		var titles = new Set();
		$('.title').each(function() {
			titles.add($(this).val());
		})
		return titles
	}

	function find_duplicate_title(){
		var titles = new Set();
		var title;
		var repeated_title;
		$('.title').each(function() {
			title = $(this).val();
			if (titles.has(title)) {
	  			$(this).addClass("border");
	  			$(this).addClass("border-danger");
	  			repeated_title = title;
			}
			else {
	  			$(this).removeClass("border");
	  			$(this).removeClass("border-danger");
			}
			titles.add(title);
		});
		return repeated_title;
	}

	function update_top_section() {
		num_checkboxes = 0;
		count = 0;
		$(".select_checkbox").each(function(){
			num_checkboxes += 1
		  	if ($(this).is(":checked")) {
		  		count += 1;
		    }
		});

		if (!count) {
			count = num_checkboxes;
		}
		$('#top_count').html('');
		var selections = [1, 3, 5, 10];
		for (var i = 1 ; i <= count; i++) {
			if (selections.includes(i)) {
				selected = 'selected';
			}
			else {
				selected = '';
			}
			$('#top_count').append("<option value='"+i+"' "+selected+">"+i+"</option>");
		}
	}

	$(document).on('change', '.select_checkbox', function(){
	    update_top_section();
	});

	function update_selectors(update_top=true) {
    	$(".title").each(function() {
	  		title = $(this).val().trim();
	  		counter = $(this).attr('counter');
	  		content = "<label for='" + title + "'>" +
            "<li class='list-group-item pb-0 pt-1'>" +
                "<a>" +
                  "<input class='select_checkbox' type='checkbox' id='" + title + "' value='"+counter+"'/>&nbsp;" + title +
                "</a>" +
            "</li>" +
          "</label>"
	  	  $('#select_paragraphs').append(content);
	  	});
	  	if (update_top) {
	  		update_top_section();
	  	}
	}

	function check_titles(update_top=true) {
		var repeated_title = find_duplicate_title();
		if (repeated_title) {
			alert("repeated");
  			var message = "Paragraph titles should be unique. `"+ repeated_title + "` is repeated.";
  			set_alert(message, "danger", "duplicate_title_alert");
  			$('#select_button').attr('disabled', true);
	  		$("#get_answer").attr("disabled", true);
	  		$('#add_new_paragraph').attr("disabled", true);
	  		$('#select_paragraphs').html('');
	  		$("#top_count").val('None');
	  		$('#top_count').attr("disabled", true);
	  		return false;
		}
		else {
			$('#alerts').html('');
  			$('#select_button').attr('disabled', false);
	  		$("#get_answer").attr("disabled", false);
	  		$('#add_new_paragraph').attr("disabled", false);
	  		$('#select_paragraphs').html('');
			update_selectors(update_top=update_top);
			return true;
		}
	}

	$(document).on('change', '.title', function() {
		check_titles();
	})

	function update_model() {
		if (!check_titles(update_top=false)) {
			return null;
		}
		var model = $('#model').val();
		if (!model) {
			set_alert("Model can not be None..", "danger");
			return null;
		}
		var message = "Getting answer";
		set_alert(message);
		$("#answer").html('');
		$('#question_error').html("");
		$("#get_answer").attr("disabled", true);

		var selected_checkboxes = [];
		var titles = [];
		var selected_titles = [];
		var paragraphs = [];
		var found_duplicate_title = false
		var found_paragraph = false;

		$(".select_checkbox").each(function(){
			if ($(this).is(":checked")) {
				selected_checkboxes.push($(this).val());
			}
		});

		for (var i = 0; i < $(".paragraph").length; i++) {
			counter = $(".title").eq(i).attr('counter');
		  	title = $(".title").eq(i).val().trim();
		  	paragraph = $(".paragraph").eq(i).val().trim();

		  	if (!paragraph && paragraph.length === 0) {
		  		continue;
		  	}
		  	if (!found_paragraph) {
		  		found_paragraph = true;
		  	}
			if (!title) {
				title = 'Paragraph ' + counter;
			}
		    if (selected_checkboxes.length === 0 || selected_checkboxes.includes(counter)) {
		  		selected_titles.push(title);
		  		paragraphs.push(paragraph);
		    }

		}

		if (!found_paragraph) {
			set_alert("At least one paragraph should have content", "danger");
			$("#get_answer").attr("disabled", false);
			return null;
		}

		data = {"paragraphs": paragraphs, "titles": selected_titles};
		if ($('#skip_paraselection').is(":checked")) {
			data['skip_paraselection'] = true;
		}
		data['top_para_count'] = parseInt($('#top_count').val());

		var question = $("#question").val().trim();

		if (!question || question.length===0) {
			set_alert("Question can not be empty", "danger");
			$("#get_answer").attr("disabled", false);
			return null;
		}

		$("#answers_section").html("");
		set_alert("Finding answer..")
		$.post('/process_question', {model:model, question: question, data: JSON.stringify(data)}
		).done(
		    function(response){
		      var predictions = response.predictions;
		      for (i=0; i<predictions.length; i++) {
		        response = predictions[i];
		        answer_div = "<div class='mt-2'>" +
		        		"<div class='row' style='background:#000;color:white'>"+
		        			"<div class='pl-2 mb-0 col-md-8'>" +
		        				response['para_title'] +
		        			"</div>"+
		        			"<div class='text-right col-md-4'>" +
		        				response['logit'] +
		        			"</div>" +
		        		"</div>" + 
						"<textarea placeholder='Answer' class='answer' disabled>" +
							response['text'] +
						"</textarea>  " +
		           "</div>";
		        $("#answers_section").append(answer_div);
		      }
		      if (predictions.length === 0) {
		      	$('#question_error').html("Unable to match any paragraph!<br>Kindly revise your question")
		      }
		      $('#alerts').html("");
			}
		).fail(
			function(response) {
				alert("Failed at process_question...")
			}
		).always(
			function(response){
				$("#get_answer").attr("disabled", false);
			}
		);
	}


	$('#get_answer').click(function() {
	    update_model();
	});

	$('#delete_all').click(function() {
		$('.paragraphs_class').each(function() {
			$(this).remove();
		});
		check_titles();
	})

	function add_new_paragraph(paragraph="", title='') {
		counter = parseInt($(".paragraph").last().attr("counter")) + 1;
		if (!counter) {
			counter = '1';
		}
		if (!title) {
			title = "Paragraph " + counter
		}
		content = "" + 
			"<div class='paragraphs_class'>" +
				"<div class='row ml-0 mr-0 mb-2'>" + 
				  	"<div class='margin col-md-11'>Title: <input type='text' class='title' placeholder='Title' counter='"+counter+"' value='"+title+"'></div>" +
		  		    "<button type='button' class='btn btn-primary col-md-1 delete_paragraph' counter='"+counter+"'>Delete</button>" +
				"</div>" +
			"<div>" +
		  	"<textarea placeholder='Enter the paragraph...' class='paragraph mb-5' counter='"+counter+"'>" + paragraph + "</textarea>"
		$("#content_section").append(content);
	}

	$('#add_new_paragraph').click(function() {
		add_new_paragraph()
		check_titles();
	});

	$(document).on("click", ".delete_paragraph", function() {
		$(this).parent().parent().remove();
		check_titles();
	});

    $('#file').on('change', function() {
        var form_data = new FormData($('#upload-file')[0]);
        $.ajax({
            type: 'POST',
            url: '/read_csv',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false
        }).done(function(response) {
        	data = response['data'];
        	$('.paragraphs_class').each(function() {
				$(this).remove();
			})
        	for (var i = 0; i < data.length; i++) {
        		add_new_paragraph(data[i][1], data[i][0])
        	}
        	$('#file').val("");
        	$('#nav-csv-tab').removeClass('active');
        	$('#nav-paragraphs-tab').addClass('active');

        	$('#nav-csv').removeClass('show');
        	$('#nav-csv').removeClass('active');

        	$('#nav-profile').addClass('show');
        	$('#nav-profile').addClass('active');
        	check_titles();

        }).fail(function(response){alert("Failed");});
    });

    $('#content_file').on('change', function() {
        var form_data = new FormData($('#upload-content-file')[0]);
        $.ajax({
            type: 'POST',
            url: '/read_content_file',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false
        }).done(function(response) {
        	data = response['data'];
        	$('.paragraphs_class').each(function() {
				$(this).remove();
			})
        	for (var i = 0; i < data.length; i++) {
        		add_new_paragraph(data[i])
        	}
        	$('#content_file').val("");
        	$('#nav-csv-tab').removeClass('active');
        	$('#nav-paragraphs-tab').addClass('active');

        	$('#nav-csv').removeClass('show');
        	$('#nav-csv').removeClass('active');

        	$('#nav-profile').addClass('show');
        	$('#nav-profile').addClass('active');
        	check_titles();

        }).fail(function(response){alert("Failed");});
    });


    $('#clear_selection').click(function() {
    	$(".select_checkbox").prop("checked", false);
    	update_top_section();
    })

});
