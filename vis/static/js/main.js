var recep_field = 22
var line_chart;

var precip_canvas = $("#precip")["0"];
var precip_ctx = precip_canvas.getContext("2d");
var precip_imgData;

var scr_lon_idx, scr_lat_id;
var scr_lon_offset = 320,
    scr_lat_offset = 66

var variable_canvas = $("#variable_map")["0"];
var variable_ctx = variable_canvas.getContext("2d");

var sensitivity_canvas = $("#sensitivity_map")["0"];
var sensitivity_ctx = sensitivity_canvas.getContext("2d");

var json;

function init() {
    // init line chart
    line_chart = c3.generate({
        bindto: "#vari_sens",
        data: {
            columns: [
                ['SLP', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ['T2', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ['HGT_500hPa', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ['HGT_250hPa', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ['U_250hPa', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ['U_200hPa', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ['SST', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        },
        axis: {
            x: {
                tick: {
                    format: function (x) { return -21 + x; }
                },
                label: "month"
            },
            y: {
                tick: {
                    format: function (y) { return y.toFixed(4); }
                },
                label: "sensitivity"
            }
        },
        legend: {
            show: true
        },
        size: {
            height: 180,
            width: 760
        },
        padding: {
            left: 60,
            bottom: 0
        },
        point: {
            show: true
        }
    });

    var url = "../static/data/ne_50m_land_0-360_transform.json"
    var request = new XMLHttpRequest();
    request.open("get", url);
    request.send(null);
    request.onload = function () {
        if (request.status == 200) {
            json = JSON.parse(request.responseText);
            console.log(json);
        }
    }

    precip_canvas.addEventListener("click", function __handler__(evt) {
        scr_lon_idx = evt.clientX;
        scr_lat_idx = evt.clientY;
        var rect = precip_canvas.getBoundingClientRect();
        scr_lon_idx -= rect.left;
        scr_lat_idx -= rect.top;
        precip_ctx.putImageData(precip_imgData, 0, 0);
        precip_ctx.fillStyle = "#0000FF";
        precip_ctx.fillRect(scr_lon_idx - 2, scr_lat_idx - 2, 4, 4);
        var lon_idx = parseInt(scr_lon_idx / 2);
        scr_lon_idx = lon_idx;
        var lat_idx = parseInt((precip_canvas.height - 1 - scr_lat_idx) / 2);
        scr_lat_idx = parseInt(scr_lat_idx / 2);
        spatial_indices = {"lat_idx": lat_idx, "lon_idx": lon_idx};
        $.ajax({
            url: "/backward_prop",
            data: spatial_indices,
            type: "POST",
            success: function(response) {
                variable_ctx.clearRect(0, 0, variable_canvas.width, variable_canvas.height);
                sensitivity_ctx.clearRect(0, 0, sensitivity_canvas.width, sensitivity_canvas.height);

                var slp = ['SLP'],
                    t2 = ['T2'],
                    ght_500hpa = ['HGT_500hPa'],
                    ght_250hpa = ['HGT_250hPa'],
                    u_250hpa = ['U_250hPa'],
                    u_200hpa = ['U_200hPa'],
                    sst = ['SST'];
                for (i = 0; i < recep_field; i++){
                    slp.push(response.grad_stats[recep_field * 0 + i]);
                    t2.push(response.grad_stats[recep_field * 1 + i]);
                    ght_500hpa.push(response.grad_stats[recep_field * 2 + i]);
                    ght_250hpa.push(response.grad_stats[recep_field * 3 + i]);
                    u_250hpa.push(response.grad_stats[recep_field * 4 + i]);
                    u_200hpa.push(response.grad_stats[recep_field * 5 + i]);
                    sst.push(response.grad_stats[recep_field * 6 + i]);
                }
                line_chart.load({
                    columns: [
                      slp, t2, ght_500hpa, ght_250hpa, u_250hpa, u_200hpa, sst
                    ]
                });
            },
            error: function(error) {
                console.log(error);
            }
      });
    });
}

function update_month(selectObject) {
    var time_idx = {"idx": selectObject.selectedIndex - 1};
    $.ajax({
        url: "/forward_prop",
        data: time_idx,
        type: "POST",
        success: function(response) {
            variable_ctx.clearRect(0, 0, variable_canvas.width, variable_canvas.height);
            sensitivity_ctx.clearRect(0, 0, sensitivity_canvas.width, sensitivity_canvas.height);
            line_chart.load({
                columns: [
                    ['SLP', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ['T2', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ['HGT_500hPa', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ['HGT_250hPa', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ['U_250hPa', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ['U_200hPa', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ['SST', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ]
            });

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
            variable_ctx.strokeStyle = '#2C0154';
            variable_ctx.strokeRect(scr_lon_offset, scr_lat_offset, parseInt(precip_canvas.width) / 2 , parseInt(precip_canvas.height) / 2);
            variable_ctx.fillStyle = "#000000";
            variable_ctx.fillRect(scr_lon_idx - 1 + scr_lon_offset, scr_lat_idx - 1 + scr_lat_offset, 2, 2);

            for (var i = 0; i < json.length; i++){
                 variable_ctx.fillRect((json[i][0] - 155.) / 160. * variable_canvas.width,
                                       (66. - json[i][1]) / 86. * variable_canvas.height, 1, 1);
            }

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
            sensitivity_ctx.strokeStyle = '#2C0154';
            sensitivity_ctx.strokeRect(scr_lon_offset, scr_lat_offset, parseInt(precip_canvas.width) / 2 , parseInt(precip_canvas.height) / 2);
            sensitivity_ctx.fillStyle = "#000000";
            sensitivity_ctx.fillRect(scr_lon_idx - 1 + scr_lon_offset, scr_lat_idx - 1 + scr_lat_offset, 2, 2);

            for (var i = 0; i < json.length; i++){
                 sensitivity_ctx.fillRect((json[i][0] - 155.) / 160. * sensitivity_canvas.width,
                                       (66. - json[i][1]) / 86. * sensitivity_canvas.height, 1, 1);
            }
        },
        error: function(error) {
            console.log(error);
        }
    });
}
