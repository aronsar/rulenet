# Rulenet

The purpose of this project is to extract rules from neural networks, by cleverly injecting boolean features into a network operating on  some data. The dataset currently used is from here: https://snap.stanford.edu/data/ego-Facebook.html

### Prerequisites

*Python 3.6
*Virtualenv
*Tensorflow 1.8 (in your virtualenv)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. Navigate to the folder which you would like to contain the project. Then execute the following commands:

```
git clone https://github.com/aronsar/rulenet.git
cd rulenet
virtualenv venv
venv\Scripts\activate #for Windows
source venv/bin/activate #for Ubuntu
pip install tensorflow=1.8 #needs an additional = sign for Windows
python main.py
tensorboard --logdir ./tf_logs
```

Now you may navigate to the webpage as directed by the console line output of tensorboard. For SSH connections, you will need to look into port forwarding in order to view the tensorboard visualizations in your browser.

## Authors

* **Aron Sarmasi** - *Initial work* - [aronsar](https://github.com/aronsar)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Joshua McCoy, for his excellent advising
* Erica and Arunpreet
* Hat tip to anyone whose code was used
* etc
