#!/usr/bin/env bash

cd ./docs;
make html;
make html;
make html;
touch _build/html/.nojekyll;
