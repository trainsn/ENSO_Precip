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

        },
        error: function(error) {
            console.log(error);
        }
    });
}
