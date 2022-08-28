# ENSO_Precip

This is the source for my summer project at Argonne National Laboratory: "A CNN-based Emulator for Climate Analysis" . 

## Play with the Visualization Explorer

###  Open the System

The system is currently deployed at ANL lambda nodes. 
For users who want to play with the system, first, run
```
ssh -J "<your_account>@logins.cels.anl.gov,<your_account>@homes.cels.anl.gov" -L 5000:localhost:5000 <your_account>@lambda2.cels.anl.gov
```
in your local machine's shell. 

Next, input "localhost:5000" into your local browser. 
I successfully ran the website with Edge on my local machine but faced some problems with Chrome (I will continue exploring the reason).

###  Select a Month for Precipitation View

Users can select a month from the drop-down box in the Precipitation View. 
Then the precipation view during the correspoding month would appear, as shown below. 
The color map used is "Viridis" from [this site](https://www.kennethmoreland.com/color-advice/).

<img src="https://github.com/trainsn/ENSO_Precip/blob/main/vis/images/update_month.png" width="40%">

###  Show Variable Sensitivity Temporal View

Users can select an arbitrary location on the map (the blue dot), and the corresponding variable sensitivity in different months is shown in the "Variable Sensitivity Temporal View". 
The sensitivity does not have units yet. 
I will try to convert it and use multiple views for different variables. 

<img src="https://github.com/trainsn/ENSO_Precip/blob/main/vis/images/temporal_sensitivity.png" width="100%">

###  Show Variable Sensitivity Temporal View

Users can select a particular variable and month in the drop-down box of the "Variable Value and Sensitivity Spatial View" to show the input variable map (left) and its sensitivity map (right). 
Please do this step after seeing the line graph in the "Variable Sensitivity Temporal View".
The color maps used for the left and right sub-figure are "Kindlmann" and "Bent Cool Warm", respectively, from [this site](https://www.kennethmoreland.com/color-advice/).

<img src="https://github.com/trainsn/ENSO_Precip/blob/main/vis/images/spatial_sensitivity.png" width="100%">
