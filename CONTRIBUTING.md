# Contributing to `delira`

If you are interested in contributing to `delira`, you will either

* implement a new feature

or 

* fix a bug.

For both types of contribution, the process is roughly the same:

1. File an issue at [this repo] and discuss 
the issue with us! Maybe we can give you some hints towards 
implementation/fixing.

2. Create your own fork of `delira`

3. In your own fork, start a new branch for the implementation of your issue. 
Make sure to include basic unittests (We know, that the current code is not 
that well tested so far, but we want to change this in future).

> **Note:** To improve readability and maintainability, [PEP8 Style](https://www.python.org/dev/peps/pep-0008/) should always be followed (no exceptions).

> **Note:** If you added a feature, you should also add it to the documentation

4. After finishing the coding part, send a pull request to 
[this repo]

5. Afterwards, have a look at your pull request since we might suggest some 
changes.


If you are not familiar with creating a Pull Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/


## Development Install

To develop `delira` on your machine, here are some tips:

1. Uninstall all existing installs of `delira`:
```
conda uninstall delira
pip uninstall delira
pip uninstall delira # run this command twice
```

2. Clone a copy of `delira` from source:

```
git clone https://github.com/justusschock/delira.git
cd delira
```

3. Install `delira` in `build develop` mode:

Install it via 

```
python setup.py build develop
```

or 

```
pip install -e .
```

This mode will symlink the python files from the current local source tree into the
python install.

Hence, if you modify a python file, you do not need to reinstall `delira` 
again and again

In case you want to reinstall, make sure that you uninstall `delira` first by running `pip uninstall delira`
and `python setup.py clean`. Then you can install in `build develop` mode again.


## Unit testing

Unittests are located under `test/`. Run the entire test suite with

```
python test/run_test.py
```

or run individual test files, like `python test/test_dummy.py`, for individual test suites.

### Better local unit tests with unittest
Testing is done with a `unittest` suite

## Writing documentation

`delira` uses [numpy style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)
for formatting docstrings. Length of line inside docstrings block must be limited to 80 characters to
fit into Jupyter documentation popups.

[this repo]: https://github.com/delira-dev/delira
