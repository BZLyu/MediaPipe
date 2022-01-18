# MediaPipe Install

https://google.github.io/mediapipe/getting_started/install.html#installing-on-macos

1. install homebrew

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

If Warning; --shallow

Update homebrew-core

```
cd /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core
git pull --unshallow
cd /usr/local/Homebrew/Library/Taps/homebrew/homebrew-cask
git pull --unshallow
```

2. Install Bazelisk.

   1. ```
      brew install bazelisk
      ```

   2. check Bazelisk

      ```
      bazel version
      ```

3. Install OpenCV and FFmpeg.

   1. ```
      $ brew install opencv@3
      
      # There is a known issue caused by the glog dependency. Uninstall glog.
      $ brew uninstall --ignore-dependencies glog
      ```

4. Make sure that Python 3 and the Python “six” library are installed.

   1. ```
      python --version
      pip3 install --user six
      ```

5. Run the [Hello World! in C++ example](https://google.github.io/mediapipe/getting_started/hello_world_cpp.html).

   1. ```
      export GLOG_logtostderr=1
      # Need bazel flag 'MEDIAPIPE_DISABLE_GPU=1' as desktop GPU is currently not supported
      $ bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
          mediapipe/examples/desktop/hello_world:hello_world
      
      # Should print:
      # Hello World!
      # Hello World!
      # Hello World!
      # Hello World!
      # Hello World!
      # Hello World!
      # Hello World!
      # Hello World!
      # Hello World!
      # Hello World!
      ```

# 配置 Pycharm:

