var parameters =  {"OmM": .12, "OmB": .0215, "h": .55, "theta": 0., "phi": 0.};
var chart0, chart1, chart2;

var canvas = $("#image")["0"];
var ctx = canvas.getContext("2d");

function init() {
  // init sliders
  var sim_slider0_data = [.120, .127, .134, .141, .148, .155];
  init_slider("#sim_slider0", "OmM", sim_slider0_data, ".3", .120);
  var sim_slider1_data = [.0215, .0220, .0225, .0230, .0235];
  init_slider("#sim_slider1", "OmB", sim_slider1_data, ".4", .0215);
  var sim_slider2_data = [.55, .60, .65, .70, .75, .80, .85];
  init_slider("#sim_slider2", "h", sim_slider2_data, ".2", .55);

  var vis_slider0_data = [0., 60., 120., 180., 240., 300., 360.];
  init_slider("#vis_slider0", "theta", vis_slider0_data, ",d", 0.);
  var vis_slider1_data = [-90., -60., -30., 0., 30., 60., 90.];
  init_slider("#vis_slider1", "phi", vis_slider1_data, ",d", 0.);

  // init line charts
  chart0 = c3.generate({
      bindto: "#sim_sens0",
      data: {
        columns: [
          ["data0", 0, 0],
        ]
      },
      axis: {
        x: {show:false},
        y: {
          tick: {
            format: function (y) { return Math.round(y); }
          },
          label: "sensitivity"
        }
      },
      legend: {
        show: false
      },
      size: {
        height: 100,
        width: 500
      },
      padding: {
        left: 80,
        bottom: 0
      },
      point: {
        show: false
      }
  });

  chart1 = c3.generate({
      bindto: '#sim_sens1',
      data: {
        columns: [
          ['data1', 0, 0],
        ]
      },
      axis: {
        x: {show:false},
        y: {
          tick: {
            format: function (y) { return Math.round(y); }
          },
          label: "sensitivity"
        }
      },
      legend: {
        show: false
      },
      size: {
        height: 100,
        width: 500
      },
      padding: {
        left: 80,
        bottom: 0
      },
      point: {
        show: false
      }
  });

  chart2 = c3.generate({
      bindto: '#sim_sens2',
      data: {
        columns: [
          ['data2', 0, 0],
        ]
      },
      axis: {
        x: {show:false},
        y: {
          tick: {
            format: function (y) { return Math.round(y); }
          },
          label: "sensitivity"
        }
      },
      legend: {
        show: false
      },
      size: {
        height: 100,
        width: 500
      },
      padding: {
        left: 80,
        bottom: 0
      },
      point: {
        show: false
      }
  });
}

function init_slider(id, name, data, format, value) {
  var slider = d3
    .sliderBottom()
    .min(d3.min(data))
    .max(d3.max(data))
    .width(400)
    .tickFormat(d3.format(format))
    .tickValues(data)
    .default(value)
    .fill("#a3d4fa")
    .on("onchange", val => {
      parameters[name] = val;

      $.ajax({
        url: "/update",
        data: parameters,
        type: "POST",
        success: function(response) {
          var canv = $("#image")["0"];
          kd.utils.draw_img_clr(canv, response.image, 2);
          var ctx = canv.getContext("2d");
        },
        error: function(error) {
          console.log(error);
        }
      });
    });

  var slider_g = d3
    .select(id)
    .append("svg")
    .attr("width", 520)
    .attr("height", 60)
    .append("g")
    .attr("transform", "translate(30,20)");

  slider_g.append("text")
    .style("font-size", "16px")
    .attr("transform", "translate(0,4)")
    .text(name);
  slider_g.append("g")
    .attr("transform", "translate(60,0)")
    .call(slider);
}

function update_month(selectObject) {
  var month_idx = {"idx": selectObject.selectedIndex - 1}
  $.ajax({
    url: "/update_month",
    data: month_idx,
    type: "POST",
    success: function(response) {
      var canvas = $("#image")["0"];
      var ctx = canvas.getContext("2d");

      var imgData = ctx.createImageData(canvas.width, canvas.height); // width x height
      var data = imgData.data;

      // copy img byte-per-byte into our ImageData
      for (var i = 0; i < canvas.height; i++) {
        for (var j = 0; j < canvas.width; j++){
            data[(i * canvas.width + j) * 4] = response.image[((canvas.height - 1 - i) * canvas.width + j) * 4]
            data[(i * canvas.width + j) * 4 + 1] = response.image[((canvas.height - 1 - i) * canvas.width + j) * 4 + 1]
            data[(i * canvas.width + j) * 4 + 2] = response.image[((canvas.height - 1 - i) * canvas.width + j) * 4 + 2]
            data[(i * canvas.width + j) * 4 + 3] = response.image[((canvas.height - 1 - i) * canvas.width + j) * 4 + 3]
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
