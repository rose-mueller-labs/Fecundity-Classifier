# need to be inside tf

python3 configure.py

bazel build --config=opt --copt=-mavx --copt=-mfma //tensorflow/tools/pip_package:build_pip_package