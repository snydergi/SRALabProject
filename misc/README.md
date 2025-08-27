This folder contains files that lack a more appropriate place within the repository.

```data_tools``` contains a variety of files developed for tasks such as confirming data frequency, extracting data from collected ```rosbag```, and plotting inference times.

```dataset_tools``` contains files related to the division of data for training, validation, and testing set creation, including ```.txt``` files holding information about each data episode.

```media``` contains images and videos primarily captured during model testing on the exoskeleton, or related to data from those tests.

```model_scripting``` contains a file used to convert ```.pth``` PyTorch model files into ```.pt``` scripts that were capable of running without any additional support code to define model structure.

```rqt_multiplot_templates``` contains ```rqt_multiplot``` plotting templates for use in live testing to observe predictions. 

```stacked_model_testing``` contains the test file used to check inference time for the stacked models.

```velocity_testing``` contains files related to the testing of filtering and calculation methods for joint velocities for models that do not include those predictions.
