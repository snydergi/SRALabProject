# About
The models in this folder are LSTMs modeling one joint, based on another joint. Due to the mirrored nature of the therapist and patient, data is linked as such: Joint 1 to Joint 3, Joint 2 to Joint 4, Joint 3 to Joint 1, and Joint 4 to Joint 2.

# Quickstart
I recommend creating a copy of the core `lstm.py` file to put into the folder of any trial you plan to run. With how the code is currently set up, this method makes the most sense for changing parameters and keeping track of alterations made to the model before training.

Within `lstm.py`, the paths to the dataset are currently set up for my CSVs, which I store in the root folder of this experiment (in this case `lstm_singleJoint`). These can be seen in the code as `'../X2_SRA_A_07-05-2024_10-39-10-mod-sync.csv'`. Be sure to update the path accordingly relative to where you store the CSV and how you have set up the file structure.

Next, ensure that all references to Joints are set to the desired Joints to study. In the code, these are under where the CSV is read in and the values are being turned into arrays to feed the data to the model.

At this point, you are ready to run the program. There are a few ways to do this. First, if you have access to a powerful central server used for training models, migrate the folders there so that your machine will not be bogged down for the near future in the process of training this (you will need to repeat the steps to create the virtual environment on this machine as well).

Then, you may run the program with `python3 lstm.py`.

An alternative to this, how I typically ran the training routine, is to call `nohup python3 lstm.py >> output.log 2>&1 &`. This runs the program, puts all terminal printouts into a file it creates next to it in the directory called `output.log` and then runs the program in the background so that you can disconnect from the remote machine without worrying about your code quitting when you disconnect.

# Testing (Validation)
As I was new to the world of Machine Learning, I did not yet quite have a grasp on all the terminology at this point in the project, and when I learned of the difference, decided to continue making progress rather than being concerned with diction. Because of this, `validation.py` is what is used to test the model after it has been trained.

`validation.py` contains code to plot a variety of graphs for interpreting the results of the model training. As with above, ensure the path to the CSV files is input accordingly, and the Joints that were read for training the model are the same as they were in `lstm.py`. 

The chunks of plotting code are labeled for what they will produce. There is capacity to create plots for the histogram of errors, true data and predicted data, error over time, and plotting the data normalized into one period or gait phase with the mean and standard deviation. Include or comment out any of these chunks as you desire.

With that, you may run the program with `python3 validation.py`.

# Parameter Search
The file `lstm_paramsearch.py` contains code to perform a grid search to try to find optimal parameters for training future iterations of the single joint LSTM. It requires the installation of `skorch` to the `venv`. In `main`, change the values of each parameter in `param_grid` to change how the search will be conducted.

Run the same as the standard `lstm.py` aside from those differences.