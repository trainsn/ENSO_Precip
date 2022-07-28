var parameters =  {"OmM": .12, "OmB": .0215, "h": .55, "theta": 0., "phi": 0.};
var chart0, chart1, chart2;

var canvas = $("#precip")["0"];
var ctx = canvas.getContext("2d");

var time_idx;

canvas.addEventListener("click", function __handler__(evt) {
    var lon_idx = evt.clientX;
    var lat_idx = evt.clientY;
    var rect = canvas.getBoundingClientRect();
    lon_idx -= rect.left;
    lat_idx -= rect.top;
    lon_idx = parseInt(lon_idx / 2);
    lat_idx = parseInt((canvas.height - 1 - lat_idx) / 2);
    time_spatial_indices = {"time_idx": time_idx["idx"], "lat_idx": lat_idx, "lon_idx": lon_idx};
    $.ajax({
        url: "/backward_prop",
        data: time_spatial_indices,
        type: "POST",
        error: function(error) {
            console.log(error);
        }
  });
});

function update_month(selectObject) {
  time_idx = {"idx": selectObject.selectedIndex - 1};
  $.ajax({
    url: "/forward_prop",
    data: time_idx,
    type: "POST",
    success: function(response) {
      var canvas = $("#precip")["0"];
      var ctx = canvas.getContext("2d");

      var imgData = ctx.createImageData(canvas.width, canvas.height); // width x height
      var data = imgData.data;

      // copy img byte-per-byte into our ImageData
      for (var i = 0; i < canvas.height; i++) {
        for (var j = 0; j < canvas.width; j++){
            data[(i * canvas.width + j) * 4] = response.image[((canvas.height - 1 - i) * canvas.width + j) * 4];
            data[(i * canvas.width + j) * 4 + 1] = response.image[((canvas.height - 1 - i) * canvas.width + j) * 4 + 1];
            data[(i * canvas.width + j) * 4 + 2] = response.image[((canvas.height - 1 - i) * canvas.width + j) * 4 + 2];
            data[(i * canvas.width + j) * 4 + 3] = response.image[((canvas.height - 1 - i) * canvas.width + j) * 4 + 3];
        }
      }

      // now we can draw our imagedata onto the canvas
      ctx.putImageData(imgData, 0, 0);
    },
    error: function(error) {
      console.log(error);
    }
  });
}

function update_psens() {
  $.ajax({
    url: "/psens",
    data: parameters,
    type: "POST",
    success: function(response) {
      var data0 = ['data0'],
          data1 = ['data1'],
          data2 = ['data2'];
      for(var i = 0; i < 10; i++) {
        for(var j = 0; j < 10; j++) {
          data0.push(Math.abs(response.psens[i][j]))
          data1.push(Math.abs(response.psens[10 + i][j]))
          data2.push(Math.abs(response.psens[20 + i][j]))
        }
      }

      chart0.load({
        columns: [
          data0
        ]
      });

      chart1.load({
        columns: [
          data1
        ]
      });

      chart2.load({
        columns: [
          data2
        ]
      });
    },
    error: function(error) {
      console.log(error);
    }
  });
}
