# RP：evaluate the uncertainty of Dynamic PET images
## "Backbone" Folder Operations

### Clustering

In the backbone folder, first, execute the TAC\_cluster.py script. This script is designed to perform clustering analysis.



### Generalized Linear Model Fitting

After completing the clustering step, run the elastic\_regression\_real\_final.py script. The primary function of this script is to fit a generalized linear model.



### Residual Permutation for Pseudo-Sample Generation

Next, execute the permute\_residual.py script. This script generates pseudo-samples through the method of residual permutation.



### Calculating Metabolic Parameters and Their Uncertainties

Upon finishing the previous steps, navigate into the Calculate\_kinetic\_parameters subfolder. Then, run the following scripts in sequence:



First, run transfer.py.

Next, run main.m.

Finally, run pi.py.

By following these operations, you can calculate the metabolic parameters of the generated pseudo-samples as well as the uncertainties associated with these parameters.



## "generate\_simulation\_data" Folder Operations

In the generate\_simulation\_data folder, execute the data\_gen.py script. This script has the capability to generate simulated Time-Activity Curves (TACs) with different noise levels. By adjusting various combinations of metabolic parameters within the script, you can generate simulated TACs corresponding to different tissues.
