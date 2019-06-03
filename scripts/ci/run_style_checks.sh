#!/usr/bin/env bash

# based onhttps://gist.github.com/MichaelCurrie/802ce28c993ff2dd632c

# find pep8 errors and ignore E402 module level import not at top of file due to logging
num_errors_before=`find . -name \*.py -exec pycodestyle --ignore=E402 {} + | wc -l`;
echo $num_errors_before;

cd "$TRAVIS_BUILD_DIR";
# try with combination of maintainer email and github token
git config user.name "Travis AutoPEP8 Fixes";
git checkout $TRAVIS_BRANCH;

# fix pep8 erros in place if possible
find . -name \*.py -exec autopep8 --recursive --aggressive --aggressive --in-place {} +;
num_errors_after=`find . -name \*.py -exec pycodestyle --ignore=E402 {} + | wc -l`;
echo $num_errors_after;

if (( $num_errors_after < $num_errors_before )); then
    git commit -a -m "PEP-8 Auto-Fix";
    git config --global push.default simple; # Push only to the current branch.  
    # Make sure to make the output quiet, or else the API token will 
    # leak!  This works because the API key can replace your password.
    git push https://$GITHUB_TOKEN@github.com/justusschock/delira.git;
fi

cd "$TRAVIS_BUILD_DIR";
# List remaining errors, which have to be fixed manually
find . -name \*.py -exec pycodestyle --ignore=E402 {} +;
