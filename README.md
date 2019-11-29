# SIGMOD Reproducibility for Paper


## A) Source code info

The **Cape** system is written in `Python` and uses [PostgreSQL](https://www.postgresql.org/) as a backend for storage. Cape is made available on [pypi](https://pypi.org/). The **Cape** package installs a library as well as a commandline tool `capexplain`. This tool can be used to mine patterns, create explanations, and to start a GUI for interactively running queries, specifying questions, and browsing patterns and explanation

- Repository: https://github.com/IITDBGroup/cape
- Programming Language: Python
- Additional Programming Language info: we are requiring Python3. Tested versions are Python 3.6 and Python 3.8.
- Required libraries/packages: `tkinter` which requires a system package to be installed (see below)


## B)  Datasets info

We used two real world datasets in the experiments:
- Publication dataset extracted from DBLP: [https://dblp.uni-trier.de/](https://dblp.uni-trier.de/)
- Crime dataset from the Chicago open data portal: [https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2)

In the experiments we utilized several subsets of these datasets. We provide a docker image with Postgres containing all of these datasets. Please install docker TODO on your machine and run the command below to fetch the image.

~~~sh
docker pull iitdbgroup/cape-experimentsTODO
~~~

You can create a container from this image using

~~~sh
docker run --name cape-postgres -d -p 5433:5432 iitdbgroup/cape-experiments
~~~

To test the container you can connect to the database and test it using:

~~~sh
docker exec -ti cape-postgres psql TODO
~~~

## C) Hardware Info

All runtime experiments were executed on a server with the following specs:

| Element          | Description                                                                   |
|------------------|-------------------------------------------------------------------------------|
| CPU              | 2 x AMD Opteron(tm) Processor 4238, 3.3Ghz                                    |
| Caches (per CPU) | L1 (288KiB), L2 (6 MiB), L3 (6MiB)                                            |
| Memory           | 128GB (DDR3 1333MHz)                                                          |
| RAID Controller  | LSI Logic / Symbios Logic MegaRAID SAS 2108 [Liberator] (rev 05), 512MB cache |
| RAID Config      | 4 x 1TB, configured as RAID 5                                                 |
| Disks            | 4 x 1TB 7.2K RPM Near-Line SAS 6Gbps (DELL CONSTELLATION ES.3)                |


## D) Installation and Setup

### Install Cape

Please follow these instructions to install the system and datasets for reproducibility. Please see below for an standard installation with pip.

#### Prerequisites ####

Cape requires python 3 and uses python's [tkinter](https://docs.python.org/3/library/tkinter.html) for its graphical UI. For example, on ubuntu you can install the prerequisites with:

~~~shell
sudo apt-get install python3 python3-pip python3-tk
~~~

#### Clone git repository

Please clone the Cape git repository and check out the `sigmod-reproducibility` branch. This branch contains Cape as well as scripts for running experiments and plotting results.

~~~shell
git clone git@github.com:IITDBGroup/cape.git capexplain
git checkout sigmod-reproducibility
cd capexplain
~~~

#### Build and Install Cape

As mentioned before, Cape is written in Python. We recommend creating a python3 [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). There are several ways to do that. Here we illustrate one. First enter the directory in which you cloned the capexplain git repository.

~~~shell
cd capexplain
~~~

Create the virtual environment.

~~~shell
python3 -m venv env
~~~

Activate the environment:

~~~shell
source ./env/bin/activate
~~~

Update `setuptools`:

~~~shell
python3 -m pip update setuptools
~~~

Install Cape:

~~~shell
python3 setup.py install
~~~

##### Test the installation

You can run

~~~shell
capexplain help
~~~

This should produce an output like this:

~~~shell
$capexplain help
capexplain COMMAND [OPTIONS]:
	explain unusually high or low aggregation query results.

AVAILABLE COMMANDS:

mine                          - Mining patterns that hold for a relation (necessary preprocessing step for generating explanations.
explain                       - Generate explanations for an aggregation result (patterns should have been mined upfront using mine).
stats                         - Extracting statistics from database collected during previous mining executions.
help                          - Show general or command specific help.
gui                           - Open the Cape graphical explanation explorer.
~~~

##### Use Docker (Alternative)

For convenience we also provide a docker image:

- TODO

### Install Postgres + load database ###

We provide a docker image with a Postgres database that contains the datasets used in the experiments. If you do not have docker, please install it:

- on mac: [https://docs.docker.com/docker-for-mac/install/](https://docs.docker.com/docker-for-mac/install/)

here: First pull the image from dockerhub:

~~~shell
docker pull iitdbgroup/2019-sigmod-reproducibility-cape-postgres
~~~

To start a container and forward its port to your local machine run the command shown below. Postgres will be available at port `5440`. Using these settings the container will be deleted once it is stopped.

~~~shell
docker run -d -p 5432:5440 --name capepostgres iitdbgroup/2019-sigmod-reproducibility-cape-postgres
~~~

To test the postgres container run:

~~~shell
docker exec -ti mypostgres psql -U postgres postgres
psql (10.0)
Type "help" for help.

postgres=#
~~~

This will connect to the Postgres instance using Postgre's commandline client `psql`. You can quit the client using `\q`.

### Run Experiments

In our experiments we evaluated three things:

- performance of the offline pattern mining algorithm
- performance of the online explanation generation algorithm
- quality of the generated explanations

#### Pattern mining

- TODO finding the right parameters
- TODO generating the subset tables

#### Explanation generation

- TODO materialize the subset pattern tables
- TODO setup the script
To test if the experiment environment has been setup correctly, run
~~~shell
bash perf_exp_crime_small.sh
~~~
to see if the code runs and plots `expl_crime_numpat.pdf` and `expl_crime_numatt.pdf` generated.

To reproduce the result:
~~~shell
bash perf_exp_crime.sh
bash perf_exp_dblp.sh
~~~

The result of Figure 6 (a) is in `expl_DBLP_numpat.pdf`; Figure 6 (b) is in `expl_crime_numpat.pdf`; Figure 6 (c) is in `expl_crime_numatt.pdf`.

#### Explanation Quality

- TODO materialize the subset pattern tables

To reproduce the result:
~~~shell
bash qual_exp_crime.sh
bash qual_exp_dblp.sh
~~~

The results for Table 3 and Table 4 are in `output_dblp.txt`, and the results for Table 5 are in `output_crime.txt`.

### Suggestions and Instructions for Alternative Experiments

For convenience, we provide the single script that runs all experiments. Creating explanations for outliers in cape consists of two steps. There is an offline mining phase that detects patterns in a dataset and an online explanation generation phase that uses the patterns to create an explanation for a user questions. To run different parameter settings, you can use the commandline client to run these phases (`capexplain COMMAND -help` lists all options that are available for a particular commeand, e.g., `mine`). Furthermore, we provide a GUI for exploring explanations. Feel free to use it for generating explanations for additional queries / user questions not covered in the experiments.


#### Pattern mining

- TODO explain how to run the algorithm with different parameters, giving some suggestions

#### Explanation generation and Explanation Quality

- TODO how to subsample the tables (create script), how to generate explanations, how to use the GUI

# Appendix
## Cape Usage ##

Cape provides a single binary `capexplain` that support multiple subcommands. The general form is:

~~~shell
capexplain COMMAND [OPTIONS]
~~~

Options are specific to each subcommand. Use `capexplain help` to see a list of supported commands and `capexplain help COMMAND` get more detailed help for a subcommand.

### Overview ###

Cape currently only supports PostgreSQL as a backend database (version 9 or higher). To use Cape to explain an aggregation outlier, you first have to let cape find patterns for the table over which you are aggregating. This an offline step that only has to be executed only once for each table (unless you want to re-run pattern mining with different parameter settings). Afterwards, you can either use the commandline or Cape's UI to request explanations for an outlier in an aggregation query result.

### Mining Patterns ###

Use `capexplain mine [OPTIONS]` to mine patterns. Cape will store the discovered patterns in the database. The "mined" patterns will be stored in a created schema called `pattern`, and the pattern tables generated after running `mine` command are `pattern.{target_table}_global` and `pattern.{target_table}_local`. At the minimum you have to tell Cape how to connect to the database you want to use and which table it should generate patterns for. Run `capexplain help mine` to get a list of all supported options for the mine command. The options needed to specify the target table and database connection are:

~~~shell
-h ,--host <arg>               - database connection host IP address (DEFAULT: 127.0.0.1)
-u ,--user <arg>               - database connection user (DEFAULT: postgres)
-p ,--password <arg>           - database connection password
-d ,--db <arg>                 - database name (DEFAULT: postgres)
-P ,--port <arg>               - database connection port (DEFAULT: 5432)
-t ,--target-table <arg>       - mine patterns for this table
~~~

For instance, if you run a postgres server locally (default) with user `postgres` (default), password `test`, and want to mine patterns for a table `employees` in database `mydb`, then run:

~~~shell
capexplain mine -p test -d mydb -t employees
~~~

#### Mining algorithm parameters ####

Cape's mining algorithm takes the following arguments:

~~~shell
--gof-const <arg>              - goodness-of-fit threshold for constant regression (DEFAULT: 0.1)
--gof-linear <arg>             - goodness-of-fit threshold for linear regression (DEFAULT: 0.1)
--confidence <arg>             - global confidence threshold
-r ,--regpackage <arg>         - regression analysis package to use {'statsmodels', 'sklearn'} (DEFAULT: statsmodels)
--local-support <arg>          - local support threshold (DEFAULT: 10)
--global-support <arg>         - global support thresh (DEFAULT: 100)
-f ,--fd-optimizations <arg>   - activate functional dependency detection and optimizations (DEFAULT: False)
-a ,--algorithm <arg>          - algorithm to use for pattern mining {'naive', 'cube', 'share_grp', 'optimized'} (DEFAULT: optimized)
--show-progress <arg>          - show progress meters (DEFAULT: True)
--manual-config                - manually configure numeric-like string fields (treat fields as string or numeric?) (DEFAULT: False)

~~~

#### Running our "crime" data example ####

We included a subset of the "Chicago Crime" dataset (https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/)
in our repository for user to play with. To import this dataset in your postgres databse, under `/testdb` directory, run the following command template:

~~~shell
psql -h <host> -U <user name> -d <local database name where you want to store our example table> < ~/cape/testdb/crime_demonstration.sql
~~~
then run the `capexplain` commands accordingly to explore this example.

### Explaining Outliers ###

To explain an aggregation outlier use `capexplain explain [OPTIONS]`.

~~~shell
-l ,--log <arg>                - select log level {DEBUG,INFO,WARNING,ERROR} (DEFAULT: ERROR)
--help                         - show this help message
-h ,--host <arg>               - database connection host IP address (DEFAULT: 127.0.0.1)
-u ,--user <arg>               - database connection user (DEFAULT: postgres)
-p ,--password <arg>           - database connection password
-d ,--db <arg>                 - database name (DEFAULT: postgres)
-P ,--port <arg>               - database connection port (DEFAULT: 5432)
--ptable <arg>                 - table storing aggregate regression patterns
--qtable <arg>                 - table storing aggregation query result
--ufile <arg>                  - file storing user question
-o ,--ofile <arg>              - file to write output to
-a ,--aggcolumn <arg>          - column that was input to the aggregation function
~~~
for `explain` option, besides the common options, user should give `--ptable`,the `pattern.{target_table}` and `--qtable`, the `target_table`. Also, we currently only allow user pass question through a `.txt` file, user need to put the question in the following format:

~~~shell
attribute1, attribute 2, attribute3...., direction
value1,value2,value3...., high/low
~~~
please refer to `input.txt` to look at an example.


### Starting the Explanation Explorer GUI ###

Cape comes with a graphical UI for running queries, selecting outliers of interest, and exploring patterns that are relevant for an outlier and browsing explanations generated by the system. You need to specify the Postgres server to connect to. The explorer can only generate explanations for queries over tables for which patterns have mined beforehand using `capexplain mine`.
Here is our demo video : (https://www.youtube.com/watch?v=gWqhIUrcwz8)

~~~shell
$ capexplain help gui
capexplain gui [OPTIONS]:
	Open the Cape graphical explanation explorer.

SUPPORTED OPTIONS:
-l ,--log <arg>                - select log level {DEBUG,INFO,WARNING,ERROR} (DEFAULT: ERROR)
--help                         - show this help message
-h ,--host <arg>               - database connection host IP address (DEFAULT: 127.0.0.1)
-u ,--user <arg>               - database connection user (DEFAULT: postgres)
-p ,--password <arg>           - database connection password
-d ,--db <arg>                 - database name (DEFAULT: postgres)
-P ,--port <arg>               - database connection port (DEFAULT: 5432)
~~~

For instance, if you run a postgres server locally (default) with user `postgres` (default), password `test`, and database `mydb`, then run:

~~~shell
capexplain gui -p test -d mydb
~~~

## Links ##

Cape is developed by researchers at Illinois Institute of Technology and Duke University. For more information and publications see the Cape project page [http://www.cs.iit.edu/~dbgroup/projects/cape.html](http://www.cs.iit.edu/~dbgroup/projects/cape.html).
