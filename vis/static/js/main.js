var parameters =  {"OmM": .12, "OmB": .0215, "h": .55, "theta": 0., "phi": 0.};
var chart0, chart1, chart2;

var precip_canvas = $("#precip")["0"];
var precip_ctx = precip_canvas.getContext("2d");
var precip_imgData;

precip_canvas.addEventListener("click", function __handler__(evt) {
    var lon_idx = evt.clientX;
    var lat_idx = evt.clientY;
    var rect = precip_canvas.getBoundingClientRect();
    lon_idx -= rect.left;
    lat_idx -= rect.top;
    precip_ctx.putImageData(precip_imgData, 0, 0);
    precip_ctx.fillStyle = "#0000FF";
    precip_ctx.fillRect(lon_idx - 2, lat_idx - 2, 4, 4);
    lon_idx = parseInt(lon_idx / 2);
    lat_idx = parseInt((precip_canvas.height - 1 - lat_idx) / 2);
    spatial_indices = {"lat_idx": lat_idx, "lon_idx": lon_idx};
    $.ajax({
        url: "/backward_prop",
        data: spatial_indices,
        type: "POST",
        error: function(error) {
            console.log(error);
        }
  });
});

function update_month(selectObject) {
    var time_idx = {"idx": selectObject.selectedIndex - 1};
    $.ajax({
        url: "/forward_prop",
        data: time_idx,
        type: "POST",
        success: function(response) {
            precip_imgData = precip_ctx.createImageData(precip_canvas.width, precip_canvas.height); // width x height
            var data = precip_imgData.data;

            // copy img byte-per-byte into our ImageData
            for (var i = 0; i < precip_canvas.height; i++) {
                for (var j = 0; j < precip_canvas.width; j++){
                    data[(i * precip_canvas.width + j) * 4] =
                        response.image[((precip_canvas.height - 1 - i) * precip_canvas.width + j) * 4];
                    data[(i * precip_canvas.width + j) * 4 + 1] =
                        response.image[((precip_canvas.height - 1 - i) * precip_canvas.width + j) * 4 + 1];
                    data[(i * precip_canvas.width + j) * 4 + 2] =
                        response.image[((precip_canvas.height - 1 - i) * precip_canvas.width + j) * 4 + 2];
                    data[(i * precip_canvas.width + j) * 4 + 3] =
                        response.image[((precip_canvas.height - 1 - i) * precip_canvas.width + j) * 4 + 3];
                }
            }

            // now we can draw our imagedata onto the canvas
            precip_ctx.putImageData(precip_imgData, 0, 0);
        },
        error: function(error) {
            console.log(error);
        }
    });
}

function update_variable_time() {
    var variable_select = document.getElementById("variable_name");
    var variable_idx = variable_select.selectedIndex - 1;
    var variable_name = variable_select.value
    var relative_month_select = document.getElementById("relative_month");
    var relative_month_idx = relative_month_select.selectedIndex;
    var vari_relamonth_idx = {"variable_idx": variable_idx, "variable_name": variable_name, "relamonth_idx": relative_month_idx};
    $.ajax({
        url: "/retrieve_variable_time",
        data: vari_relamonth_idx,
        type: "POST",
        success: function(response) {
            var variable_canvas = $("#variable_map")["0"];
            var variable_ctx = variable_canvas.getContext("2d");

            var variable_imgData = variable_ctx.createImageData(variable_canvas.width, variable_canvas.height); // width x height
            var variable_data = variable_imgData.data;

            // copy img byte-per-byte into our ImageData
            for (var i = 0; i < variable_canvas.height; i++) {
                for (var j = 0; j < variable_canvas.width; j++){
                    variable_data[(i * variable_canvas.width + j) * 4] =
                        response.value_image[((variable_canvas.height - 1 - i) * variable_canvas.width + j) * 4];
                    variable_data[(i * variable_canvas.width + j) * 4 + 1] =
                        response.value_image[((variable_canvas.height - 1 - i) * variable_canvas.width + j) * 4 + 1];
                    variable_data[(i * variable_canvas.width + j) * 4 + 2] =
                        response.value_image[((variable_canvas.height - 1 - i) * variable_canvas.width + j) * 4 + 2];
                    variable_data[(i * variable_canvas.width + j) * 4 + 3] =
                        response.value_image[((variable_canvas.height - 1 - i) * variable_canvas.width + j) * 4 + 3];
                }
            }

            // now we can draw our imagedata onto the canvas
            variable_ctx.putImageData(variable_imgData, 0, 0);

            var sensitivity_canvas = $("#sensitivity_map")["0"];
            var sensitivity_ctx = sensitivity_canvas.getContext("2d");

            var sensitivity_imgData = sensitivity_ctx.createImageData(sensitivity_canvas.width, sensitivity_canvas.height); // width x height
            var sensitivity_data = sensitivity_imgData.data;

            // copy img byte-per-byte into our ImageData
            for (var i = 0; i < sensitivity_canvas.height; i++) {
                for (var j = 0; j < sensitivity_canvas.width; j++){
                    sensitivity_data[(i * sensitivity_canvas.width + j) * 4] =
                        response.sensitivity_image[((sensitivity_canvas.height - 1 - i) * sensitivity_canvas.width + j) * 4];
                    sensitivity_data[(i * sensitivity_canvas.width + j) * 4 + 1] =
                        response.sensitivity_image[((sensitivity_canvas.height - 1 - i) * sensitivity_canvas.width + j) * 4 + 1];
                    sensitivity_data[(i * sensitivity_canvas.width + j) * 4 + 2] =
                        response.sensitivity_image[((sensitivity_canvas.height - 1 - i) * sensitivity_canvas.width + j) * 4 + 2];
                    sensitivity_data[(i * sensitivity_canvas.width + j) * 4 + 3] =
                        response.sensitivity_image[((sensitivity_canvas.height - 1 - i) * sensitivity_canvas.width + j) * 4 + 3];
                }
            }

            // now we can draw our imagedata onto the canvas
            sensitivity_ctx.putImageData(sensitivity_imgData, 0, 0);
        },
        error: function(error) {
            console.log(error);
        }
    });
}
